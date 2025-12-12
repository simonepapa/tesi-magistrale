export const number_of_people: Record<string, number> = {
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

class MinMaxScaler {
  min: number;
  max: number;
  featureRange: [number, number];

  constructor(featureRange: [number, number] = [0, 1]) {
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

export const analyze_quartieri = (
  articles: any[],
  quartieri_data: any[],
  geojson_data: any,
  selected_crimes: string[],
  weightsForArticles: boolean = true,
  weightsForPeople: boolean = false,
  minmaxScaler: boolean = true
) => {
  const scaler = new MinMaxScaler([0, 100]);

  const crimes: Record<string, number> = {
    omicidio: 0,
    omicidio_colposo: 0,
    omicidio_stradale: 0,
    tentato_omicidio: 0,
    furto: 0,
    rapina: 0,
    violenza_sessuale: 0,
    aggressione: 0,
    spaccio: 0,
    truffa: 0,
    estorsione: 0,
    contrabbando: 0,
    associazione_di_tipo_mafioso: 0
  };

  const weights: Record<string, number> = {
    omicidio: 1,
    omicidio_colposo: 0.7,
    omicidio_stradale: 0.8,
    tentato_omicidio: 0.9,
    furto: 0.2,
    rapina: 0.7,
    violenza_sessuale: 0.8,
    aggressione: 0.6,
    spaccio: 0.5,
    truffa: 0.3,
    estorsione: 0.6,
    contrabbando: 0.4,
    associazione_di_tipo_mafioso: 1
  };

  // Filter crimes
  let filtered_crimes = crimes;
  if (selected_crimes.length > 0) {
    filtered_crimes = Object.keys(crimes)
      .filter((key) => selected_crimes.includes(key))
      .reduce(
        (obj, key) => {
          const val = crimes[key];
          if (val !== undefined) {
            obj[key] = val;
          }
          return obj;
        },
        {} as Record<string, number>
      );
  }

  // Group by quartiere
  const groupedArticles = articles.reduce(
    (acc, article) => {
      const quartiere = article.quartiere;
      if (!acc[quartiere]) acc[quartiere] = [];
      acc[quartiere].push(article);
      return acc;
    },
    {} as Record<string, any[]>
  );

  for (const group in groupedArticles) {
    const group_df = groupedArticles[group];
    const crime_data: Record<
      string,
      { frequenza: number; crime_index: number }
    > = {};

    for (const crime in filtered_crimes) {
      crime_data[crime] = { frequenza: 0, crime_index: 0 };
    }

    // Frequency
    group_df.forEach((row: any) => {
      for (const crime in filtered_crimes) {
        if (row[crime] === 1 && crime_data[crime]) {
          crime_data[crime].frequenza += 1;
        }
      }
    });

    // Weighted risk index
    const crimini_totali = Object.values(crime_data).reduce(
      (sum, c) => sum + c.frequenza,
      0
    );
    for (const crime in filtered_crimes) {
      if (crime_data[crime] && weights[crime]) {
        const risk_index = crime_data[crime].frequenza * weights[crime];
        crime_data[crime].crime_index = risk_index;
      }
    }

    // Update quartieri_data
    const quartiereEntry = quartieri_data.find(
      (quartiere: any) => quartiere.Quartiere === group
    );
    if (quartiereEntry) {
      quartiereEntry["Peso quartiere"] = group_df.length;
      quartiereEntry["Totale crimini"] = crimini_totali;
    }

    // Risk index by quartiere
    let indice_di_rischio_totale = Object.values(crime_data).reduce(
      (sum, c) => sum + c.crime_index,
      0
    );

    if (weightsForArticles) {
      indice_di_rischio_totale = indice_di_rischio_totale / group_df.length;
    }
    if (weightsForPeople) {
      const population = number_of_people[group];
      if (population) {
        indice_di_rischio_totale = indice_di_rischio_totale / population;
      }
    }

    if (minmaxScaler) {
      const np_values = Object.values(crime_data).map((c) => c.crime_index);
      const scaled_values = scaler.fit_transform(np_values);
      let idx = 0;
      for (const crime in crime_data) {
        if (crime_data[crime]) {
          crime_data[crime].crime_index = scaled_values[idx]!;
          idx++;
        }
      }
    }

    // Add to GeoJSON
    const feature = geojson_data.features.find(
      (f: any) => f.properties.python_id === group
    );
    if (feature) {
      feature.properties.crimini = crime_data;
    }

    // Save risk index
    if (quartiereEntry) {
      if (weightsForPeople) {
        quartiereEntry["Indice di rischio"] = indice_di_rischio_totale * 10000;
      } else {
        quartiereEntry["Indice di rischio"] = indice_di_rischio_totale;
      }
    }
  }

  // Scale overall risk index
  const riskIndices = quartieri_data.map(
    (quartiere: any) => quartiere["Indice di rischio"]
  );
  const scaledRiskIndices = scaler.fit_transform(riskIndices);
  quartieri_data.forEach((quartiere: any, i: number) => {
    quartiere["Indice di rischio scalato"] = scaledRiskIndices[i];
  });

  geojson_data.weightsForArticles = weightsForArticles;
  geojson_data.weightsForPeople = weightsForPeople;
  geojson_data.minmaxScaler = minmaxScaler;

  return geojson_data;
};

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
    const crimini_totali = row["Totale crimini"];
    const crime_index = row["Indice di rischio"];
    const crime_index_scalato = row["Indice di rischio scalato"];

    statistiche_dict[quartiere] = {
      crimini_totali: crimini_totali,
      crime_index: Number(crime_index.toFixed(2)),
      crime_index_scalato: Number(crime_index_scalato.toFixed(2)),
      population: number_of_people[quartiere] || 0
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
