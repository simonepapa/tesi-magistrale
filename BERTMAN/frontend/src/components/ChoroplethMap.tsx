import { Crime, InfoQuartiere } from "../types/global";
import { Feature, GeoJsonObject } from "geojson";
import { GeoJSONOptions, Layer, LeafletMouseEvent } from "leaflet";
import { Dispatch, SetStateAction, useCallback, useMemo } from "react";
import { GeoJSON, useMap } from "react-leaflet";

type Props = {
  setInfo: Dispatch<SetStateAction<InfoQuartiere>>;
  data: GeoJsonObject | null;
  color: string;
  legendValues: number[];
};

function ChoroplethMap({ setInfo, data, color, legendValues }: Props) {
  const map = useMap();

  const colorScales: { [key: string]: string[] } = useMemo(
    () => ({
      red: ["#4a0000", "#b71c1c", "#e57373", "#ffcdd2"],
      blue: ["#0d47a1", "#1976d2", "#64b5f6", "#bbdefb"],
      green: ["#1b5e20", "#388e3c", "#81c784", "#c8e6c9"]
    }),
    []
  );

  const getColor = useCallback(
    (d: number) => {
      for (let i = 0; i < legendValues.length - 1; i++) {
        if (d >= legendValues[i] && d < legendValues[i + 1]) {
          return colorScales[color][colorScales[color].length - 1 - i];
        }
      }

      return colorScales[color][0];
    },
    [color, colorScales, legendValues]
  );

  const style = useCallback(
    (feature: Feature) => {
      return {
        fillColor: getColor(feature.properties?.crime_index_scalato),
        weight: 1,
        opacity: 1,
        color: "white",
        dashArray: "2",
        fillOpacity: 0.5
      };
    },
    [getColor]
  );

  const highlightFeature = useCallback(
    (e: LeafletMouseEvent) => {
      let crimes: Crime[] = [];

      const allCrimes = e.target.feature.properties.crimini;
      if (allCrimes !== undefined) {
        crimes = Object.keys(allCrimes).map((crimine) => {
          return {
            crime: crimine,
            index: (
              Math.round(
                (allCrimes[crimine].crime_index + Number.EPSILON) * 100
              ) / 100
            ).toFixed(2),
            frequency: allCrimes[crimine].frequenza
          };
        });
      } else {
        crimes = [
          {
            crime: "aggressione",
            index: 0,
            frequency: 0
          },
          {
            crime: "associazione_di_tipo_mafioso",
            index: 0,
            frequency: 0
          },
          {
            crime: "contrabbando",
            index: 0,
            frequency: 0
          },
          {
            crime: "estorsione",
            index: 0,
            frequency: 0
          },
          {
            crime: "furto",
            index: 0,
            frequency: 0
          },
          {
            crime: "omicidio",
            index: 0,
            frequency: 0
          },
          {
            crime: "omicidio_colposo",
            index: 0,
            frequency: 0
          },
          {
            crime: "omicidio_stradale",
            index: 0,
            frequency: 0
          },
          {
            crime: "rapina",
            index: 0,
            frequency: 0
          },
          {
            crime: "spaccio",
            index: 0,
            frequency: 0
          },
          {
            crime: "tentato_omicidio",
            index: 0,
            frequency: 0
          },
          {
            crime: "truffa",
            index: 0,
            frequency: 0
          },
          {
            crime: "violenza_sessuale",
            index: 0,
            frequency: 0
          }
        ];
      }

      setInfo({
        name: e.target.feature.properties.name,
        crime_index: e.target.feature.properties.crime_index_scalato,
        total_crimes: e.target.feature.properties.crimini_totali,
        population: e.target.feature.properties.population,
        crimes: crimes,
        poi_counts: e.target.feature.properties.poi_counts
      });

      const layer = e.target;
      layer.setStyle({
        weight: 5,
        color: "#666",
        dashArray: "",
        fillOpacity: 0.7
      });

      layer.bringToFront();
    },
    [setInfo]
  );

  const resetHighlight = useCallback(
    (e: LeafletMouseEvent) => {
      setInfo({
        name: "",
        crime_index: null,
        total_crimes: null,
        population: 0,
        crimes: []
      });

      e.target.setStyle(style(e.target.feature));
    },
    [setInfo, style]
  );

  const zoomToFeature = useCallback(
    (e: LeafletMouseEvent) => {
      map.fitBounds(e.target.getBounds());
    },
    [map]
  );

  const onEachFeature = useCallback(
    (_feature: Feature, layer: Layer) => {
      layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: zoomToFeature
      });
    },
    [highlightFeature, resetHighlight, zoomToFeature]
  );

  if (data)
    return (
      <GeoJSON
        data={data as GeoJsonObject}
        style={style as GeoJSONOptions}
        onEachFeature={onEachFeature}
      />
    );
}
export default ChoroplethMap;
