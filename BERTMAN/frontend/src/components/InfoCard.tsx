import { getCrimeName } from "../helpers/utils";
import { Crime, POICounts, SubIndices } from "../types/global";
import { Separator } from "@/components/ui/separator";

type Props = {
  name: string;
  crime_index: number | null;
  population: number;
  crimes: Crime[];
  poi_counts?: POICounts;
  sub_indices?: SubIndices;
  activeSubIndices?: {
    [key: string]: number;
  };
  showPoi?: boolean;
};

// POI labels and colors for display
const poiConfig: { key: keyof POICounts; label: string; color: string }[] = [
  { key: "bar", label: "Bar", color: "#f59e0b" },
  { key: "scommesse", label: "Scommesse", color: "#ef4444" },
  { key: "bancomat", label: "Bancomat", color: "#3b82f6" },
  { key: "stazione", label: "Stazione", color: "#22c55e" }
];

function InfoCard({
  name,
  crime_index,
  population,
  crimes,
  poi_counts,
  sub_indices,
  activeSubIndices,
  showPoi = true
}: Props) {
  const numberOfCrimes = crimes.reduce(
    (acc: number, crime: Crime) => acc + crime.frequency,
    0
  );

  const totalPoi = poi_counts
    ? poi_counts.bar +
      poi_counts.scommesse +
      poi_counts.bancomat +
      poi_counts.stazione
    : 0;

  return (
    <div className="info-card bg-foreground text-background">
      <h3 className="text-lg font-bold">{name || "Neighborhood"}</h3>
      <p>
        Crime index: {crime_index}
        <span className="text-sm">
          {" "}
          - {population.toLocaleString("it-IT")} people
        </span>
      </p>
      <p className="text-sm">
        {numberOfCrimes} cases,{" "}
        {!isNaN(Math.round((numberOfCrimes / population) * 1000))
          ? Math.round((numberOfCrimes / population) * 1000)
          : 0}{" "}
        crimes per 1000 people
      </p>
      {poi_counts && totalPoi > 0 && showPoi && (
        <>
          <Separator className="my-2" />
          <div className="mb-2">
            <p className="mb-1 text-sm font-semibold">
              POI ({totalPoi} in total)
            </p>
            <div className="flex flex-wrap gap-2">
              {poiConfig.map(({ key, label, color }) =>
                poi_counts[key] > 0 ? (
                  <div
                    key={key}
                    className="flex items-center gap-1 rounded px-2 py-0.5 text-xs"
                    style={{ backgroundColor: color + "30" }}>
                    <div
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span>
                      {label}: {poi_counts[key]}
                    </span>
                  </div>
                ) : null
              )}
            </div>
          </div>
        </>
      )}
      {sub_indices && (
        <>
          <Separator className="my-2" />
          <div className="mb-2">
            <p className="mb-1 text-sm font-semibold">Sub-Indices</p>
            <div className="flex flex-wrap gap-2 text-xs">
              <div className="rounded bg-red-500/20 px-2 py-0.5">
                Crime: {sub_indices.S_crim.toFixed(1)}
              </div>
              {activeSubIndices?.poi === 1 && sub_indices.S_poi > 0 && (
                <div className="rounded bg-orange-500/20 px-2 py-0.5">
                  POI: {sub_indices.S_poi.toFixed(1)}
                </div>
              )}
              {activeSubIndices?.socioEconomic === 1 &&
                sub_indices.S_soc > 0 && (
                  <div className="rounded bg-blue-500/20 px-2 py-0.5">
                    Socio-Eco: {sub_indices.S_soc.toFixed(1)}
                  </div>
                )}
            </div>
          </div>
        </>
      )}
      <Separator className="my-2" />
      <div className="flex gap-2 overflow-auto !p-0 xl:flex-col">
        {crimes.map((crime: Crime, index: number) => (
          <div className="flex flex-col gap-0" key={index}>
            <p className="text-base font-medium">{getCrimeName(crime.crime)}</p>
            <p className="text-sm">{crime.frequency} cases</p>
          </div>
        ))}
      </div>
    </div>
  );
}
export default InfoCard;
