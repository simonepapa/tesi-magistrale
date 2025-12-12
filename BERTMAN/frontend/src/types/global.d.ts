export type Crime = {
  crime: string;
  index: number | string;
  frequency: number;
};

export type InfoQuartiere = {
  name: string;
  crime_index: number | null;
  total_crimes: number | null;
  population: number;
  crimes: Crime[];
  weights?: { [key: string]: boolean };
  minmax: boolean;
};

export type Article = {
  id: number;
  link: string;
  title: string;
  content: string;
  date: string;
  aggressione?: number;
  aggressione_prob?: number;
  associazione_di_tipo_mafioso?: number;
  associazione_di_tipo_mafioso_prob?: number;
  contrabbando?: number;
  contrabbando_prob?: number;
  estorsione?: number;
  estorsione_prob?: number;
  furto?: number;
  furto_prob?: number;
  omicidio?: number;
  omicidio_colposo?: number;
  omicidio_colposo_prob?: number;
  omicidio_prob?: number;
  omicidio_stradale?: number;
  omicidio_stradale_prob?: number;
  quartiere?: string;
  rapina?: number;
  rapina_prob?: number;
  spaccio?: number;
  spaccio_prob?: number;
  tentato_omicidio?: number;
  tentato_omicidio_prob?: number;
  truffa?: number;
  truffa_prob?: number;
  violenza_sessuale?: number;
  violenza_sessuale_prob?: number;
  [key: string]: number;
};

export type LabeledArticle = {
  link: string;
  title: string;
  content: string;
  date: string;
  [key: string]: {
    prob: number;
    value: number;
  };
};

export type Filters = {
  crimes: {
    [key: string]: number;
  };
  quartieri: {
    [key: string]: number;
  };
  weights: {
    [key: string]: number;
  };
  scaling: {
    [key: string]: number;
  };
  dates: {
    [key: string]: Date | null;
  };
};
