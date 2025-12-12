import { colorizeSquare } from "../helpers/utils";

type Props = {
  palette: string;
  legendValues: number[];
};

function ChoroplethLegend({ palette }: Props) {
  // Define fixed interval labels
  const intervalLabels = ["81-100", "51-80", "21-50", "0-20"];

  return (
    <div className="legend-card bg-foreground text-background grid grid-cols-3 gap-2 xl:grid-cols-1">
      {intervalLabels.map((label: string, index: number) => (
        <div className="flex items-center gap-2" key={label}>
          <div
            className="h-4 w-4"
            style={{ backgroundColor: colorizeSquare(index, palette) }}></div>
          <p>{label}</p>
        </div>
      ))}
    </div>
  );
}
export default ChoroplethLegend;
