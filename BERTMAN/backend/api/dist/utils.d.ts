export declare const number_of_people: Record<string, number>;
export declare const crimeWeights: Record<string, number>;
export declare const poiTypes: readonly ["bar", "scommesse", "bancomat", "stazione"];
type PoiType = (typeof poiTypes)[number];
export declare const crimePoiAffinity: Record<string, Record<PoiType, number>>;
export interface AnalysisOptions {
    enableCrimeSubIndex?: boolean;
    enableSocioEconomicSubIndex?: boolean;
    enablePoiSubIndex?: boolean;
    enableEventSubIndex?: boolean;
}
export declare const analyze_quartieri: (articles: any[], quartieri_data: any[], geojson_data: any, selected_crimes: string[], poiCountsByQuartiere?: Record<string, Record<string, number>>, options?: AnalysisOptions) => any;
export declare const calculate_statistics: (quartieri_data: any[], geojson_data: any) => any;
export {};
//# sourceMappingURL=utils.d.ts.map