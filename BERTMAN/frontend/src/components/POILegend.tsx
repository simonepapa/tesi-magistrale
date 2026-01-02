type Props = {
  visible: boolean;
};

const poiTypes = [
  { tipo: "bar", label: "Bar", color: "#f59e0b" },
  { tipo: "scommesse", label: "Scommesse", color: "#ef4444" },
  { tipo: "bancomat", label: "Bancomat", color: "#3b82f6" },
  { tipo: "stazione", label: "Stazione", color: "#22c55e" }
];

function POILegend({ visible }: Props) {
  if (!visible) return null;

  return (
    <div className="poi-legend bg-foreground text-background rounded p-3 shadow-lg">
      <h4 className="mb-2 text-sm font-semibold">Punti di Interesse</h4>
      <div className="flex flex-col gap-1.5">
        {poiTypes.map(({ tipo, label, color }) => (
          <div key={tipo} className="flex items-center gap-2">
            <div
              className="h-3 w-3 rounded-full border-2 border-white"
              style={{ backgroundColor: color }}
            />
            <span className="">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default POILegend;
