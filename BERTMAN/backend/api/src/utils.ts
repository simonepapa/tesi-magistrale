// Population data for each quartiere (for demographic normalization)
export const number_of_people: Record<string, number> = {
  "bari-vecchia_san-nicola": 5827,
  carbonara: 23971,
  carrassi: 32789,
  "ceglie-del-campo": 7467,
  japigia: 29903,
  liberta: 35629,
  loseto: 3008,
  madonnella: 10959,
  murat: 23141,
  "palese-macchie": 13644,
  picone: 40236,
  "san-paolo": 29912,
  "san-pasquale": 19726,
  "santo-spirito": 12076,
  stanic: 4288,
  "torre-a-mare": 6257,
  "san-girolamo_fesca": 15685
};

// Population density (residents/km2)
export const quartieri_density: Record<string, number> = {
  "bari-vecchia_san-nicola": 6471,
  carbonara: 1877,
  carrassi: 10733,
  "ceglie-del-campo": 1159,
  japigia: 1948,
  liberta: 19571,
  loseto: 390,
  madonnella: 19758,
  murat: 16581,
  "palese-macchie": 1342,
  picone: 2749,
  "san-paolo": 3379,
  "san-pasquale": 4784,
  "santo-spirito": 1440,
  stanic: 398,
  "torre-a-mare": 1148,
  "san-girolamo_fesca": 3036
};

// School dropout rate (18-24 years) (%)
export const quartieri_school_dropout: Record<string, number> = {
  "bari-vecchia_san-nicola": 26.6,
  carbonara: 13.0,
  carrassi: 8.1,
  "ceglie-del-campo": 14.2,
  japigia: 12.8,
  liberta: 22.0,
  loseto: 8.8,
  madonnella: 15.0,
  murat: 11.7,
  "palese-macchie": 9.2,
  picone: 7.0,
  "san-paolo": 22.2,
  "san-pasquale": 8.0,
  "santo-spirito": 10.2,
  stanic: 17.9,
  "torre-a-mare": 10.7,
  "san-girolamo_fesca": 13.1
};

// Unemployment rate (15+ years) (%)
export const quartieri_unemployment: Record<string, number> = {
  "bari-vecchia_san-nicola": 13.9,
  carbonara: 11.6,
  carrassi: 10.1,
  "ceglie-del-campo": 12.1,
  japigia: 10.7,
  liberta: 13.2,
  loseto: 11.5,
  madonnella: 11.1,
  murat: 9.7,
  "palese-macchie": 11.2,
  picone: 9.3,
  "san-paolo": 13.1,
  "san-pasquale": 9.8,
  "santo-spirito": 10.7,
  stanic: 13.2,
  "torre-a-mare": 11.7,
  "san-girolamo_fesca": 10.7
};

// Potential economic distress (%)
export const quartieri_economic_distress: Record<string, number> = {
  "bari-vecchia_san-nicola": 4.4,
  carbonara: 2.8,
  carrassi: 2.1,
  "ceglie-del-campo": 2.3,
  japigia: 1.9,
  liberta: 3.2,
  loseto: 2.8,
  madonnella: 2.7,
  murat: 2.3,
  "palese-macchie": 2.5,
  picone: 1.6,
  "san-paolo": 3.0,
  "san-pasquale": 1.8,
  "santo-spirito": 2.6,
  stanic: 4.1,
  "torre-a-mare": 3.0,
  "san-girolamo_fesca": 2.8
};

// Crime weights based on maximum penalty from Italian Penal Code (0-100 scale)
export const crimeWeights: Record<string, number> = {
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
export const poiTypes = ["bar", "scommesse", "bancomat", "stazione"] as const;
type PoiType = (typeof poiTypes)[number];

// Affinity matrix between crimes and POI types (how much a POI "attracts" a crime)
// Values range from 0 (no affinity) to 1 (high affinity)
export const crimePoiAffinity: Record<string, Record<PoiType, number>> = {
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
  min: number;
  max: number;
  featureRange: [number, number];

  constructor(featureRange: [number, number] = [0, 100]) {
    this.featureRange = featureRange;
    this.min = 0;
    this.max = 1;
  }

  fit(data: number[]) {
    this.min = Math.min(...data);
    this.max = Math.max(...data);
  }

  transform(data: number[]): number[] {
    const [minRange, maxRange] = this.featureRange;
    if (this.max === this.min) return data.map(() => minRange);
    return data.map(
      (x) =>
        ((x - this.min) / (this.max - this.min)) * (maxRange - minRange) +
        minRange
    );
  }

  fit_transform(data: number[]): number[] {
    this.fit(data);
    return this.transform(data);
  }
}

// SUB-INDEX 1: Crime History Sub-index (S_crim)
// Formula: S_crim,j = Σ(n_i,j * w_i) / P_j
// Population normalization is always applied
function calculateCrimeSubIndex(
  articles: any[],
  quartiere: string,
  selectedCrimes: string[],
  normalizeByArticles: boolean = false
): {
  subIndex: number;
  crimeData: Record<string, { frequenza: number; crime_index: number }>;
} {
  // Filter articles for this quartiere
  const quartiereArticles = articles.filter((a) => a.quartiere === quartiere);

  // Initialize crime data
  const crimeData: Record<string, { frequenza: number; crime_index: number }> =
    {};
  for (const crime of selectedCrimes) {
    crimeData[crime] = { frequenza: 0, crime_index: 0 };
  }

  // Count crime frequencies
  quartiereArticles.forEach((article: any) => {
    for (const crime of selectedCrimes) {
      if (article[crime] === 1 && crimeData[crime]) {
        crimeData[crime]!.frequenza += 1;
      }
    }
  });

  // Calculate weighted sum: Σ(n_i,j * w_i)
  let weightedSum = 0;
  for (const crime of selectedCrimes) {
    const freq = crimeData[crime]?.frequenza || 0;
    const weight = crimeWeights[crime] || 0;
    const crimeIndex = freq * weight;
    if (crimeData[crime]) {
      crimeData[crime]!.crime_index = crimeIndex;
    }
    weightedSum += crimeIndex;
  }

  // Always normalize by population: S_crim,j = Σ(n_i,j * w_i) / P_j
  const population = number_of_people[quartiere] || 1;
  let subIndex = weightedSum / population;

  // Optional: Normalize by number of articles
  if (normalizeByArticles) {
    const articleCount = quartiereArticles.length || 1;
    subIndex = subIndex / articleCount;
  }

  return { subIndex, crimeData };
}

// SUB-INDEX 2: POI Sub-index (S_poi)
// Formula: S_poi,j = Σ_k [ p_k,j * Σ_i (w_i * α_i,k) ]
function calculatePoiSubIndex(
  poiCounts: Record<PoiType, number>,
  selectedCrimes: string[]
): number {
  let subIndex = 0;

  for (const poiType of poiTypes) {
    const poiCount = poiCounts[poiType] || 0;

    // Calculate danger score for this POI type: Σ_i (w_i * α_i,k)
    let poiDangerScore = 0;
    for (const crime of selectedCrimes) {
      const weight = crimeWeights[crime] || 0;
      const affinity = crimePoiAffinity[crime]?.[poiType] || 0;
      poiDangerScore += weight * affinity;
    }

    // Multiply by POI count and add to total
    subIndex += poiCount * poiDangerScore;
  }

  return subIndex;
}

// SUB-INDEX 3: Socio-economic Sub-index (S_soc)
// Formula: S_soc,j = density + dropout + unemployment + distress
function calculateSocioEconomicSubIndex(quartiere: string): number {
  // Get raw socio-economic indicators for this quartiere
  const density = quartieri_density[quartiere] || 0;
  const dropout = quartieri_school_dropout[quartiere] || 0;
  const unemployment = quartieri_unemployment[quartiere] || 0;
  const distress = quartieri_economic_distress[quartiere] || 0;

  // Raw sub-index: sum of all indicators (will be normalized later)
  return density + dropout + unemployment + distress;
}

// SUB-INDEX 4: Temporal Events Sub-index (S_event)
// Formula: S_event,j,t = Σ_e (I_e,j,t * δ_e)
function calculateEventSubIndex(
  _quartiere: string,
  _timestamp?: Date,
  _activeEvents?: Array<{ eventId: string; impactCoefficient: number }>
): number {
  // To be developed as soon as data is available
  return 0;
}

// FINAL CRI FORMULA
// CRI_j,t = (K_crim*Ŝ_crim + K_soc*Ŝ_soc + K_poi*Ŝ_poi + K_event*Ŝ_event) / (K_crim + K_soc + K_poi + K_event)
interface SubIndexCoefficients {
  K_crim: number; // 0 or 1 to enable/disable crime sub-index
  K_soc: number; // 0 or 1 to enable/disable socio-economic sub-index
  K_poi: number; // 0 or 1 to enable/disable POI sub-index
  K_event: number; // 0 or 1 to enable/disable event sub-index
}

function calculateFinalCRI(
  normalizedSubIndices: {
    S_crim: number;
    S_soc: number;
    S_poi: number;
    S_event: number;
  },
  coefficients: SubIndexCoefficients
): number {
  const numerator =
    coefficients.K_crim * normalizedSubIndices.S_crim +
    coefficients.K_soc * normalizedSubIndices.S_soc +
    coefficients.K_poi * normalizedSubIndices.S_poi +
    coefficients.K_event * normalizedSubIndices.S_event;

  const denominator =
    coefficients.K_crim +
    coefficients.K_soc +
    coefficients.K_poi +
    coefficients.K_event;

  if (denominator === 0) return 0;

  return numerator / denominator;
}

// MAIN ANALYSIS FUNCTION
export interface AnalysisOptions {
  // Sub-index activation coefficients (K values)
  enableCrimeSubIndex?: boolean;
  enableSocioEconomicSubIndex?: boolean;
  enablePoiSubIndex?: boolean;
  enableEventSubIndex?: boolean;
  weightsForArticles?: boolean;
}

export const analyze_quartieri = (
  articles: any[],
  quartieri_data: any[],
  geojson_data: any,
  selected_crimes: string[],
  poiCountsByQuartiere?: Record<string, Record<string, number>>,
  options?: AnalysisOptions
) => {
  const scaler = new MinMaxScaler([0, 100]);

  // Determine which sub-indices are enabled
  const coefficients: SubIndexCoefficients = {
    K_crim: options?.enableCrimeSubIndex !== false ? 1 : 0, // Default: enabled
    K_soc: options?.enableSocioEconomicSubIndex === true ? 1 : 0, // Default: disabled (no data)
    K_poi: options?.enablePoiSubIndex !== false && poiCountsByQuartiere ? 1 : 0, // Enabled if POI data available
    K_event: options?.enableEventSubIndex === true ? 1 : 0 // Default: disabled (no data)
  };

  // Get list of quartieri to process
  const quartieriList = quartieri_data.map((q) => q.Quartiere);

  // Storage for raw sub-indices (before normalization)
  const rawSubIndices: Record<
    string,
    {
      S_crim: number;
      S_soc: number;
      S_poi: number;
      S_event: number;
      crimeData: Record<string, { frequenza: number; crime_index: number }>;
      totalCrimes: number;
      articleCount: number;
    }
  > = {};

  // STEP 1: Calculate raw sub-indices for each quartiere
  for (const quartiere of quartieriList) {
    // Calculate crime sub-index (always normalized by population, optionally by articles)
    const { subIndex: S_crim_raw, crimeData } = calculateCrimeSubIndex(
      articles,
      quartiere,
      selected_crimes,
      options?.weightsForArticles
    );

    // Calculate POI sub-index
    const poiCounts = poiCountsByQuartiere?.[quartiere] as
      | Record<PoiType, number>
      | undefined;
    const S_poi_raw = poiCounts
      ? calculatePoiSubIndex(poiCounts, selected_crimes)
      : 0;

    // Calculate socio-economic sub-index
    const S_soc_raw = calculateSocioEconomicSubIndex(quartiere);

    // Calculate event sub-index
    const S_event_raw = calculateEventSubIndex(quartiere);

    // Count total crimes and articles
    const totalCrimes = Object.values(crimeData).reduce(
      (sum, c) => sum + c.frequenza,
      0
    );
    const articleCount = articles.filter(
      (a) => a.quartiere === quartiere
    ).length;

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

  // Normalize Socio-Economic sub-index components separately for equal weighting
  const densities = quartieriList.map((q) => quartieri_density[q] || 0);
  const dropouts = quartieriList.map((q) => quartieri_school_dropout[q] || 0);
  const unemployments = quartieriList.map(
    (q) => quartieri_unemployment[q] || 0
  );
  const distresses = quartieriList.map(
    (q) => quartieri_economic_distress[q] || 0
  );

  const normDensities = scaler.fit_transform(densities);
  const normDropouts = scaler.fit_transform(dropouts);
  const normUnemployments = scaler.fit_transform(unemployments);
  const normDistresses = scaler.fit_transform(distresses);

  // Apply normalized values
  quartieriList.forEach((q, i) => {
    if (rawSubIndices[q]) {
      rawSubIndices[q]!.S_crim = normalizedCrim[i] || 0;
      rawSubIndices[q]!.S_poi = normalizedPoi[i] || 0;
      // S_soc is the average of the 4 normalized components (each 0-100)
      rawSubIndices[q]!.S_soc =
        ((normDensities[i] || 0) +
          (normDropouts[i] || 0) +
          (normUnemployments[i] || 0) +
          (normDistresses[i] || 0)) /
        4;
    }
  });

  // STEP 3: Calculate final CRI for each quartiere
  for (const quartiere of quartieriList) {
    const subIndices = rawSubIndices[quartiere];
    if (!subIndices) continue;

    const finalCRI = calculateFinalCRI(
      {
        S_crim: subIndices.S_crim,
        S_soc: subIndices.S_soc,
        S_poi: subIndices.S_poi,
        S_event: subIndices.S_event
      },
      coefficients
    );

    // Update quartieri_data
    const quartiereEntry = quartieri_data.find(
      (q: any) => q.Quartiere === quartiere
    );
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
    const feature = geojson_data.features.find(
      (f: any) => f.properties.python_id === quartiere
    );
    if (feature) {
      // Normalize individual crime indices within quartiere for visualization
      const crimeIndices = Object.values(subIndices.crimeData).map(
        (c) => c.crime_index
      );
      const normalizedCrimeIndices = scaler.fit_transform(crimeIndices);
      let idx = 0;
      for (const crime of Object.keys(subIndices.crimeData)) {
        subIndices.crimeData[crime]!.crime_index =
          normalizedCrimeIndices[idx] || 0;
        idx++;
      }
      feature.properties.crimini = subIndices.crimeData;

      // Add sub-indices to feature properties for InfoCard display
      feature.properties.sub_indices = {
        S_crim: subIndices.S_crim,
        S_poi: subIndices.S_poi,
        S_soc: subIndices.S_soc,
        S_event: subIndices.S_event
      };
    }
  }

  // STEP 4: Scale overall risk indices
  const riskIndices = quartieri_data.map(
    (q: any) => q["Indice di rischio"] || 0
  );
  const scaledRiskIndices = scaler.fit_transform(riskIndices);
  quartieri_data.forEach((quartiere: any, i: number) => {
    quartiere["Indice di rischio scalato"] = scaledRiskIndices[i];
  });

  // Store metadata in GeoJSON
  geojson_data.subIndexCoefficients = coefficients;
  geojson_data.formula =
    "CRI = (K_crim*S_crim + K_soc*S_soc + K_poi*S_poi + K_event*S_event) / (K_crim + K_soc + K_poi + K_event)";

  return geojson_data;
};

// STATISTICS CALCULATION
export const calculate_statistics = (
  quartieri_data: any[],
  geojson_data: any
) => {
  const statistiche_dict: Record<string, any> = {};

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
      population: number_of_people[quartiere] || 0,
      // Sub-indices for explainability
      sub_indices: {
        S_crim: Number((row["S_crim"] || 0).toFixed(2)),
        S_poi: Number((row["S_poi"] || 0).toFixed(2)),
        S_soc: Number((row["S_soc"] || 0).toFixed(2)),
        S_event: Number((row["S_event"] || 0).toFixed(2))
      }
    };
  });

  geojson_data.features.forEach((feature: any) => {
    const python_id = feature.properties.python_id;
    if (statistiche_dict[python_id]) {
      Object.assign(feature.properties, statistiche_dict[python_id]);
    }
  });

  return geojson_data;
};
