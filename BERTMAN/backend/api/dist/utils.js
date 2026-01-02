"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculate_statistics = exports.analyze_quartieri = exports.crimePoiAffinity = exports.poiTypes = exports.crimeWeights = exports.number_of_people = void 0;
// Population data for each quartiere (for demographic normalization)
exports.number_of_people = {
    "bari-vecchia_san-nicola": 5726,
    carbonara: 22625,
    carrassi: 34248,
    "ceglie-del-campo": 5018,
    japigia: 30153,
    liberta: 38701,
    loseto: 7580,
    madonnella: 10680,
    murat: 29638,
    "palese-macchie": 7315,
    picone: 40225,
    "san-paolo": 27990,
    "san-pasquale": 18313,
    "santo-spirito": 1858,
    stanic: 4489,
    "torre-a-mare": 5070,
    "san-girolamo_fesca": 4721
};
// Crime weights based on maximum penalty from Italian Penal Code (0-100 scale)
exports.crimeWeights = {
    omicidio: 100,
    associazione_di_tipo_mafioso: 95,
    tentato_omicidio: 90,
    violenza_sessuale: 85,
    rapina: 80,
    estorsione: 75,
    omicidio_stradale: 70,
    spaccio: 65,
    aggressione: 50,
    omicidio_colposo: 45,
    furto: 40,
    contrabbando: 30,
    truffa: 20
};
// POI types
exports.poiTypes = ["bar", "scommesse", "bancomat", "stazione"];
// Affinity matrix between crimes and POI types (how much a POI "attracts" a crime)
// Values range from 0 (no affinity) to 1 (high affinity)
exports.crimePoiAffinity = {
    omicidio: { bar: 0.1, scommesse: 0.3, bancomat: 0.1, stazione: 0.2 },
    associazione_di_tipo_mafioso: {
        bar: 0.2,
        scommesse: 0.8,
        bancomat: 0.1,
        stazione: 0.1
    },
    tentato_omicidio: { bar: 0.3, scommesse: 0.4, bancomat: 0.1, stazione: 0.2 },
    violenza_sessuale: { bar: 0.7, scommesse: 0.1, bancomat: 0.1, stazione: 0.8 },
    rapina: { bar: 0.3, scommesse: 0.9, bancomat: 1.0, stazione: 0.5 },
    estorsione: { bar: 0.5, scommesse: 0.7, bancomat: 0.1, stazione: 0.1 },
    omicidio_stradale: { bar: 0.6, scommesse: 0.0, bancomat: 0.0, stazione: 0.0 },
    spaccio: { bar: 0.4, scommesse: 0.2, bancomat: 0.1, stazione: 0.8 },
    aggressione: { bar: 0.7, scommesse: 0.5, bancomat: 0.2, stazione: 0.6 },
    omicidio_colposo: { bar: 0.1, scommesse: 0.1, bancomat: 0.0, stazione: 0.1 },
    furto: { bar: 0.2, scommesse: 0.2, bancomat: 0.8, stazione: 0.7 },
    contrabbando: { bar: 0.2, scommesse: 0.4, bancomat: 0.1, stazione: 0.5 },
    truffa: { bar: 0.2, scommesse: 0.6, bancomat: 0.7, stazione: 0.3 }
};
// MinMax Scaler class for normalization (always applied)
class MinMaxScaler {
    min;
    max;
    featureRange;
    constructor(featureRange = [0, 100]) {
        this.featureRange = featureRange;
        this.min = 0;
        this.max = 1;
    }
    fit(data) {
        this.min = Math.min(...data);
        this.max = Math.max(...data);
    }
    transform(data) {
        const [minRange, maxRange] = this.featureRange;
        if (this.max === this.min)
            return data.map(() => minRange);
        return data.map((x) => ((x - this.min) / (this.max - this.min)) * (maxRange - minRange) +
            minRange);
    }
    fit_transform(data) {
        this.fit(data);
        return this.transform(data);
    }
}
// SUB-INDEX 1: Crime History Sub-index (S_crim)
// Formula: S_crim,j = Σ(n_i,j * w_i) / P_j
// Population normalization is always applied
function calculateCrimeSubIndex(articles, quartiere, selectedCrimes) {
    // Filter articles for this quartiere
    const quartiereArticles = articles.filter((a) => a.quartiere === quartiere);
    // Initialize crime data
    const crimeData = {};
    for (const crime of selectedCrimes) {
        crimeData[crime] = { frequenza: 0, crime_index: 0 };
    }
    // Count crime frequencies
    quartiereArticles.forEach((article) => {
        for (const crime of selectedCrimes) {
            if (article[crime] === 1 && crimeData[crime]) {
                crimeData[crime].frequenza += 1;
            }
        }
    });
    // Calculate weighted sum: Σ(n_i,j * w_i)
    let weightedSum = 0;
    for (const crime of selectedCrimes) {
        const freq = crimeData[crime]?.frequenza || 0;
        const weight = exports.crimeWeights[crime] || 0;
        const crimeIndex = freq * weight;
        if (crimeData[crime]) {
            crimeData[crime].crime_index = crimeIndex;
        }
        weightedSum += crimeIndex;
    }
    // Always normalize by population: S_crim,j = Σ(n_i,j * w_i) / P_j
    const population = exports.number_of_people[quartiere] || 1;
    const subIndex = weightedSum / population;
    return { subIndex, crimeData };
}
// SUB-INDEX 2: POI Sub-index (S_poi)
// Formula: S_poi,j = Σ_k [ p_k,j * Σ_i (w_i * α_i,k) ]
function calculatePoiSubIndex(poiCounts, selectedCrimes) {
    let subIndex = 0;
    for (const poiType of exports.poiTypes) {
        const poiCount = poiCounts[poiType] || 0;
        // Calculate danger score for this POI type: Σ_i (w_i * α_i,k)
        let poiDangerScore = 0;
        for (const crime of selectedCrimes) {
            const weight = exports.crimeWeights[crime] || 0;
            const affinity = exports.crimePoiAffinity[crime]?.[poiType] || 0;
            poiDangerScore += weight * affinity;
        }
        // Multiply by POI count and add to total
        subIndex += poiCount * poiDangerScore;
    }
    return subIndex;
}
// SUB-INDEX 3: Socio-economic Sub-index (S_soc)
// Formula: S_soc,j = α1*X_u,j + α2*X_g,j + α3*X_d,j + α4*X_s,j
function calculateSocioEconomicSubIndex(_quartiere, _socioEconomicData) {
    // To be developed as soon as data is available
    return 0;
}
// SUB-INDEX 4: Temporal Events Sub-index (S_event)
// Formula: S_event,j,t = Σ_e (I_e,j,t * δ_e)
function calculateEventSubIndex(_quartiere, _timestamp, _activeEvents) {
    // To be developed as soon as data is available
    return 0;
}
function calculateFinalCRI(normalizedSubIndices, coefficients) {
    const numerator = coefficients.K_crim * normalizedSubIndices.S_crim +
        coefficients.K_soc * normalizedSubIndices.S_soc +
        coefficients.K_poi * normalizedSubIndices.S_poi +
        coefficients.K_event * normalizedSubIndices.S_event;
    const denominator = coefficients.K_crim +
        coefficients.K_soc +
        coefficients.K_poi +
        coefficients.K_event;
    if (denominator === 0)
        return 0;
    return numerator / denominator;
}
const analyze_quartieri = (articles, quartieri_data, geojson_data, selected_crimes, poiCountsByQuartiere, options) => {
    const scaler = new MinMaxScaler([0, 100]);
    // Determine which sub-indices are enabled
    const coefficients = {
        K_crim: options?.enableCrimeSubIndex !== false ? 1 : 0, // Default: enabled
        K_soc: options?.enableSocioEconomicSubIndex === true ? 1 : 0, // Default: disabled (no data)
        K_poi: options?.enablePoiSubIndex !== false && poiCountsByQuartiere ? 1 : 0, // Enabled if POI data available
        K_event: options?.enableEventSubIndex === true ? 1 : 0 // Default: disabled (no data)
    };
    // Get list of quartieri to process
    const quartieriList = quartieri_data.map((q) => q.Quartiere);
    // Storage for raw sub-indices (before normalization)
    const rawSubIndices = {};
    // STEP 1: Calculate raw sub-indices for each quartiere
    for (const quartiere of quartieriList) {
        // Calculate crime sub-index (always normalized by population)
        const { subIndex: S_crim_raw, crimeData } = calculateCrimeSubIndex(articles, quartiere, selected_crimes);
        // Calculate POI sub-index
        const poiCounts = poiCountsByQuartiere?.[quartiere];
        const S_poi_raw = poiCounts
            ? calculatePoiSubIndex(poiCounts, selected_crimes)
            : 0;
        // Calculate socio-economic sub-index
        const S_soc_raw = calculateSocioEconomicSubIndex(quartiere);
        // Calculate event sub-index
        const S_event_raw = calculateEventSubIndex(quartiere);
        // Count total crimes and articles
        const totalCrimes = Object.values(crimeData).reduce((sum, c) => sum + c.frequenza, 0);
        const articleCount = articles.filter((a) => a.quartiere === quartiere).length;
        rawSubIndices[quartiere] = {
            S_crim: S_crim_raw,
            S_soc: S_soc_raw,
            S_poi: S_poi_raw,
            S_event: S_event_raw,
            crimeData,
            totalCrimes,
            articleCount
        };
    }
    // STEP 2: Normalize each sub-index using MinMax scaling (0-100)
    // Normalize crime sub-index
    const crimValues = quartieriList.map((q) => rawSubIndices[q]?.S_crim || 0);
    const normalizedCrim = scaler.fit_transform(crimValues);
    // Normalize POI sub-index
    const poiValues = quartieriList.map((q) => rawSubIndices[q]?.S_poi || 0);
    const normalizedPoi = scaler.fit_transform(poiValues);
    // Apply normalized values
    quartieriList.forEach((q, i) => {
        if (rawSubIndices[q]) {
            rawSubIndices[q].S_crim = normalizedCrim[i] || 0;
            rawSubIndices[q].S_poi = normalizedPoi[i] || 0;
        }
    });
    // STEP 3: Calculate final CRI for each quartiere
    for (const quartiere of quartieriList) {
        const subIndices = rawSubIndices[quartiere];
        if (!subIndices)
            continue;
        const finalCRI = calculateFinalCRI({
            S_crim: subIndices.S_crim,
            S_soc: subIndices.S_soc,
            S_poi: subIndices.S_poi,
            S_event: subIndices.S_event
        }, coefficients);
        // Update quartieri_data
        const quartiereEntry = quartieri_data.find((q) => q.Quartiere === quartiere);
        if (quartiereEntry) {
            quartiereEntry["Peso quartiere"] = subIndices.articleCount;
            quartiereEntry["Totale crimini"] = subIndices.totalCrimes;
            quartiereEntry["Indice di rischio"] = finalCRI;
            // Store sub-indices for transparency/explainability
            quartiereEntry["S_crim"] = subIndices.S_crim;
            quartiereEntry["S_poi"] = subIndices.S_poi;
            quartiereEntry["S_soc"] = subIndices.S_soc;
            quartiereEntry["S_event"] = subIndices.S_event;
        }
        // Update GeoJSON feature
        const feature = geojson_data.features.find((f) => f.properties.python_id === quartiere);
        if (feature) {
            // Normalize individual crime indices within quartiere for visualization
            const crimeIndices = Object.values(subIndices.crimeData).map((c) => c.crime_index);
            const normalizedCrimeIndices = scaler.fit_transform(crimeIndices);
            let idx = 0;
            for (const crime of Object.keys(subIndices.crimeData)) {
                subIndices.crimeData[crime].crime_index =
                    normalizedCrimeIndices[idx] || 0;
                idx++;
            }
            feature.properties.crimini = subIndices.crimeData;
        }
    }
    // STEP 4: Scale overall risk indices
    const riskIndices = quartieri_data.map((q) => q["Indice di rischio"] || 0);
    const scaledRiskIndices = scaler.fit_transform(riskIndices);
    quartieri_data.forEach((quartiere, i) => {
        quartiere["Indice di rischio scalato"] = scaledRiskIndices[i];
    });
    // Store metadata in GeoJSON
    geojson_data.subIndexCoefficients = coefficients;
    geojson_data.formula =
        "CRI = (K_crim*S_crim + K_soc*S_soc + K_poi*S_poi + K_event*S_event) / (K_crim + K_soc + K_poi + K_event)";
    return geojson_data;
};
exports.analyze_quartieri = analyze_quartieri;
// STATISTICS CALCULATION
const calculate_statistics = (quartieri_data, geojson_data) => {
    const statistiche_dict = {};
    quartieri_data.forEach((row) => {
        if (!row["Peso quartiere"]) {
            row["Peso quartiere"] = 1.0;
        }
        const quartiere = row["Quartiere"];
        const crimini_totali = row["Totale crimini"] || 0;
        const crime_index = row["Indice di rischio"] || 0;
        const crime_index_scalato = row["Indice di rischio scalato"] || 0;
        statistiche_dict[quartiere] = {
            crimini_totali: crimini_totali,
            crime_index: Number(crime_index.toFixed(2)),
            crime_index_scalato: Number(crime_index_scalato.toFixed(2)),
            population: exports.number_of_people[quartiere] || 0,
            // Sub-indices for explainability
            sub_indices: {
                S_crim: Number((row["S_crim"] || 0).toFixed(2)),
                S_poi: Number((row["S_poi"] || 0).toFixed(2)),
                S_soc: Number((row["S_soc"] || 0).toFixed(2)),
                S_event: Number((row["S_event"] || 0).toFixed(2))
            }
        };
    });
    geojson_data.features.forEach((feature) => {
        const python_id = feature.properties.python_id;
        if (statistiche_dict[python_id]) {
            Object.assign(feature.properties, statistiche_dict[python_id]);
        }
    });
    return geojson_data;
};
exports.calculate_statistics = calculate_statistics;
//# sourceMappingURL=utils.js.map