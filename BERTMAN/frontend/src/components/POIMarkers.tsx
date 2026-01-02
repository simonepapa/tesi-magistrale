import { POI } from "../types/global";
import { LatLngExpression } from "leaflet";
import { useMemo } from "react";
import { CircleMarker, Pane, Popup, Tooltip } from "react-leaflet";

type Props = {
  poi: POI[];
};

// Color mapping for different POI types
const poiColors: { [key: string]: string } = {
  bar: "#f59e0b",
  scommesse: "#ef4444",
  bancomat: "#3b82f6",
  stazione: "#22c55e",
  default: "#6b7280"
};

// Italian labels for POI types
const poiLabels: { [key: string]: string } = {
  bar: "Bar",
  scommesse: "Scommesse",
  bancomat: "Bancomat",
  stazione: "Stazione"
};

function POIMarkers({ poi }: Props) {
  const markers = useMemo(() => {
    return poi.map((p, index) => {
      const position: LatLngExpression = [
        p.geometry.coordinates[1],
        p.geometry.coordinates[0]
      ];
      const color = poiColors[p.tipo_poi] || poiColors.default;
      const label = poiLabels[p.tipo_poi] || p.tipo_poi;

      return (
        <CircleMarker
          key={`poi-${index}-${p.quartiere_id}-${p.tipo_poi}`}
          center={position}
          radius={6}
          pathOptions={{
            fillColor: color,
            fillOpacity: 0.8,
            color: "#ffffff",
            weight: 2,
            opacity: 1
          }}>
          <Tooltip
            direction="top"
            offset={[0, -5]}
            opacity={0.9}
            className="poi-tooltip">
            <span className="font-medium">{label}</span>
          </Tooltip>
          <Popup className="z-[750]">
            <div className="text-sm">
              <p className="font-semibold">{label}</p>
              <p className="text-gray-600">
                Quartiere:{" "}
                {p.quartiere_id.replace(/-/g, " ").replace(/_/g, " ")}
              </p>
            </div>
          </Popup>
        </CircleMarker>
      );
    });
  }, [poi]);

  return (
    <Pane name="poi-markers" style={{ zIndex: 650 }}>
      {markers}
    </Pane>
  );
}

export default POIMarkers;
