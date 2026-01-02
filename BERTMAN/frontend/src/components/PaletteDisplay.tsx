import { colorizeSquare } from "../helpers/utils";

type Props = {
  palette: string;
};

function PaletteDisplay({ palette }: Props) {
  return (
    <div className="flex flex-col">
      <div
        className="h-4 w-4"
        style={{ backgroundColor: colorizeSquare(3, palette) }}></div>
      <div
        className="h-4 w-4"
        style={{ backgroundColor: colorizeSquare(2, palette) }}></div>
      <div
        className="h-4 w-4"
        style={{ backgroundColor: colorizeSquare(1, palette) }}></div>
      <div
        className="h-4 w-4"
        style={{ backgroundColor: colorizeSquare(0, palette) }}></div>
    </div>
  );
}
export default PaletteDisplay;
