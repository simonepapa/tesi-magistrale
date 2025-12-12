import { Filters } from "../types/global";
import PaletteDisplay from "./PaletteDisplay";
import TileDisplay from "./TileDisplay";
import { Button } from "./ui/button";
import { Checkbox } from "./ui/checkbox";
import { DateRangePicker } from "./ui/date-range-picker";
import { Label } from "./ui/label";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";
import { Separator } from "./ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { Info } from "lucide-react";
import { Dispatch, SetStateAction, useCallback } from "react";

type Props = {
  palette: string;
  setPalette: Dispatch<SetStateAction<string>>;
  tile: string;
  setTile: Dispatch<SetStateAction<string>>;
  filters: Filters;
  setFilters: Dispatch<SetStateAction<Filters>>;
  fetchData: () => void;
  startDate: Date | null;
  endDate: Date | null;
  setEndDate: Dispatch<SetStateAction<Date | null>>;
  setStartDate: Dispatch<SetStateAction<Date | null>>;
  handleResetDate: () => void;
};

function DashboardLeft({
  palette,
  setPalette,
  tile,
  setTile,
  filters,
  setFilters,
  fetchData,
  startDate,
  endDate,
  setEndDate,
  setStartDate,
  handleResetDate
}: Props) {
  const handleTileChange = (style: string) => {
    setTile(style);
  };

  const handlePaletteChange = (color: string) => {
    setPalette(color);
  };

  const handleFiltersChange = (crime: string, type: keyof Filters) => {
    // State copy
    const filtersCopy: Filters = { ...filters };

    filtersCopy[type][crime] = filtersCopy[type][crime] === 1 ? 0 : 1;

    setFilters({
      ...filters,
      [type]: filtersCopy[type]
    });
  };

  const handleResetFilters = () => {
    setFilters({
      crimes: {
        omicidio: 1,
        omicidio_colposo: 1,
        omicidio_stradale: 1,
        tentato_omicidio: 1,
        furto: 1,
        rapina: 1,
        violenza_sessuale: 1,
        aggressione: 1,
        spaccio: 1,
        truffa: 1,
        estorsione: 1,
        contrabbando: 1,
        associazione_di_tipo_mafioso: 1
      },
      quartieri: {
        "bari-vecchia_san-nicola": 1,
        carbonara: 1,
        carrassi: 1,
        "ceglie-del-campo": 1,
        japigia: 1,
        liberta: 1,
        loseto: 1,
        madonnella: 1,
        murat: 1,
        "palese-macchie": 1,
        picone: 1,
        "san-paolo": 1,
        "san-pasquale": 1,
        "santo-spirito": 1,
        stanic: 1,
        "torre-a-mare": 1,
        "san-girolamo_fesca": 1
      },
      weights: {
        num_of_articles: 1,
        num_of_people: 0
      },
      scaling: {
        minmax: 1
      },
      dates: {
        startDate: null,
        endDate: null
      }
    });
    handleResetDate();
  };

  const handleApply = useCallback(() => {
    fetchData();
    setFilters({
      ...filters,
      dates: {
        startDate,
        endDate
      }
    });
  }, [endDate, fetchData, filters, setFilters, startDate]);

  return (
    <div className="dashboard-left bg-background flex flex-col gap-4 xl:pr-4">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-4">
          <h2 className="text-3xl font-bold">Data filters</h2>
          <Button size="sm" onClick={handleResetFilters}>
            Reset to default
          </Button>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <label className="text-lg font-medium">
                Filter by date range
              </label>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full">
                    <Info className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">
                    Select a range to limit the date range of the news. Note
                    that this filter only works if you select both dates
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>

            <div className="flex flex-col gap-2 lg:flex-row lg:gap-4">
              <DateRangePicker
                value={{
                  from: startDate || undefined,
                  to: endDate || undefined
                }}
                onChange={(range) => {
                  setStartDate(range?.from || null);
                  setEndDate(range?.to || null);
                }}
                disableFuture={true}
                className="lg:flex-1"
              />
            </div>
          </div>

          <Separator className="my-1" />

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <label className="text-lg font-medium">Filter by crimes</label>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full">
                    <Info className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">
                    Select which crimes to show. Please note that this will
                    change the index's value
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex flex-row flex-wrap gap-3">
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.omicidio === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("omicidio", "crimes")
                  }
                />
                <span className="text-sm">Murder</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.omicidio_colposo === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("omicidio_colposo", "crimes")
                  }
                />
                <span className="text-sm">Manslaughter</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.omicidio_stradale === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("omicidio_stradale", "crimes")
                  }
                />
                <span className="text-sm">Road homicide</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.tentato_omicidio === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("tentato_omicidio", "crimes")
                  }
                />
                <span className="text-sm">Attempted murder</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.furto === 1}
                  onCheckedChange={() => handleFiltersChange("furto", "crimes")}
                />
                <span className="text-sm">Theft</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.rapina === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("rapina", "crimes")
                  }
                />
                <span className="text-sm">Robber</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.violenza_sessuale === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("violenza_sessuale", "crimes")
                  }
                />
                <span className="text-sm">Sexual violence</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.aggressione === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("aggressione", "crimes")
                  }
                />
                <span className="text-sm">Assault</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.spaccio === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("spaccio", "crimes")
                  }
                />
                <span className="text-sm">Drug trafficking</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.truffa === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("truffa", "crimes")
                  }
                />
                <span className="text-sm">Fraud</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.estorsione === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("estorsione", "crimes")
                  }
                />
                <span className="text-sm">Extortion</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.contrabbando === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("contrabbando", "crimes")
                  }
                />
                <span className="text-sm">Smuggling</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.crimes.associazione_di_tipo_mafioso === 1}
                  onCheckedChange={() =>
                    handleFiltersChange(
                      "associazione_di_tipo_mafioso",
                      "crimes"
                    )
                  }
                />
                <span className="text-sm">Mafia-type association</span>
              </label>
            </div>
          </div>

          <Separator className="my-1" />

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <label className="text-lg font-medium">
                Filter by neighborhood
              </label>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full">
                    <Info className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">
                    Select which neighborhoods to show. Please note that this
                    will change the index's value
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex flex-row flex-wrap gap-3">
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["bari-vecchia_san-nicola"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("bari-vecchia_san-nicola", "quartieri")
                  }
                />
                <span className="text-sm">Bari Vecchia - San Nicola</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["carbonara"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("carbonara", "quartieri")
                  }
                />
                <span className="text-sm">Carbonara</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["carrassi"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("carrassi", "quartieri")
                  }
                />
                <span className="text-sm">Carrassi</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["ceglie-del-campo"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("ceglie-del-campo", "quartieri")
                  }
                />
                <span className="text-sm">Ceglie del Campo</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["japigia"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("japigia", "quartieri")
                  }
                />
                <span className="text-sm">Japigia</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["liberta"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("liberta", "quartieri")
                  }
                />
                <span className="text-sm">Libertà</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["loseto"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("loseto", "quartieri")
                  }
                />
                <span className="text-sm">Loseto</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["madonnella"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("madonnella", "quartieri")
                  }
                />
                <span className="text-sm">Madonnella</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["murat"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("murat", "quartieri")
                  }
                />
                <span className="text-sm">Murat</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["palese-macchie"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("palese-macchie", "quartieri")
                  }
                />
                <span className="text-sm">Palese - Macchie</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["picone"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("picone", "quartieri")
                  }
                />
                <span className="text-sm">Picone</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["san-paolo"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("san-paolo", "quartieri")
                  }
                />
                <span className="text-sm">San Paolo</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["san-pasquale"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("san-pasquale", "quartieri")
                  }
                />
                <span className="text-sm">San Pasquale</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["santo-spirito"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("santo-spirito", "quartieri")
                  }
                />
                <span className="text-sm">
                  Santo Spirito - San Pio - Catino
                </span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["stanic"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("stanic", "quartieri")
                  }
                />
                <span className="text-sm">Stanic</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["torre-a-mare"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("torre-a-mare", "quartieri")
                  }
                />
                <span className="text-sm">Torre a mare</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  checked={filters.quartieri["san-girolamo_fesca"] === 1}
                  onCheckedChange={() =>
                    handleFiltersChange("san-girolamo_fesca", "quartieri")
                  }
                />
                <span className="text-sm">San Girolamo - Fesca</span>
              </label>
            </div>
          </div>

          <Separator className="my-1" />

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <label className="text-lg font-medium">Weights</label>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full">
                    <Info className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">
                    Select the weights that will afflict the overall index score
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex flex-row flex-wrap gap-3">
              <div className="flex items-center">
                <label className="flex cursor-pointer items-center gap-2">
                  <Checkbox
                    checked={filters.weights["num_of_articles"] === 1}
                    onCheckedChange={() =>
                      handleFiltersChange("num_of_articles", "weights")
                    }
                  />
                  <span className="text-sm">Number of articles</span>
                </label>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full">
                      <Info className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-sm">
                      Divides the crime index by the total number of articles
                      for that neighborhood in the database
                    </p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="flex items-center">
                <label className="flex cursor-pointer items-center gap-2">
                  <Checkbox
                    checked={filters.weights["num_of_people"] === 1}
                    onCheckedChange={() =>
                      handleFiltersChange("num_of_people", "weights")
                    }
                  />
                  <span className="text-sm">Number of people</span>
                </label>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full">
                      <Info className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-sm">
                      Divides the crime index by the number of people for that
                      neighborhood. <br />
                      <b>WARNING</b>: to prevent the index from being rounded to
                      0, it will be multiplied by 10.000
                    </p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>

          <Separator className="my-1" />

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-1">
              <label className="text-lg font-medium">Scaling</label>
            </div>
            <div className="flex flex-row flex-wrap gap-3">
              <div className="flex items-center">
                <label className="flex cursor-pointer items-center gap-2">
                  <Checkbox
                    checked={filters.scaling["minmax"] === 1}
                    onCheckedChange={() =>
                      handleFiltersChange("minmax", "scaling")
                    }
                  />
                  <span className="text-sm">Minmax scaler</span>
                </label>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full">
                      <Info className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-sm">
                      Scale the indices into a [0,100] range. Note that this way
                      there will always be an element with index 0 (min) and an
                      element with index 100 (max). The colors of the map will
                      be based on the scaled value.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </div>
        <Button
          className="!mx-auto !mt-4 w-full sm:!mr-0 sm:!ml-auto"
          onClick={handleApply}>
          Apply
        </Button>
      </div>

      <Separator className="my-1" />

      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold">Map style</h2>
        <div className="flex flex-col gap-2">
          <Label className="text-lg font-medium">Layers palette</Label>
          <RadioGroup
            value={palette}
            onValueChange={handlePaletteChange}
            className="flex flex-row gap-8">
            <div className="flex items-center gap-2">
              <RadioGroupItem value="red" id="palette-red" />
              <Label htmlFor="palette-red" className="cursor-pointer">
                <PaletteDisplay palette="red" />
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="blue" id="palette-blue" />
              <Label htmlFor="palette-blue" className="cursor-pointer">
                <PaletteDisplay palette="blue" />
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="green" id="palette-green" />
              <Label htmlFor="palette-green" className="cursor-pointer">
                <PaletteDisplay palette="green" />
              </Label>
            </div>
          </RadioGroup>
        </div>
        <div className="flex flex-col gap-2">
          <Label className="mb-2 text-lg font-medium">Tile style</Label>
          <RadioGroup
            value={tile}
            onValueChange={handleTileChange}
            className="flex flex-row gap-2">
            <div className="flex items-center gap-2">
              <RadioGroupItem
                value="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                id="tile-base"
              />
              <Label htmlFor="tile-base" className="cursor-pointer">
                <TileDisplay style="base" />
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem
                value="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                id="tile-dark"
              />
              <Label htmlFor="tile-dark" className="cursor-pointer">
                <TileDisplay style="dark" />
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem
                value="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                id="tile-light"
              />
              <Label htmlFor="tile-light" className="cursor-pointer">
                <TileDisplay style="light" />
              </Label>
            </div>
          </RadioGroup>
        </div>
      </div>
    </div>
  );
}
export default DashboardLeft;
