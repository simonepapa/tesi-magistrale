import ChoroplethLegend from "../components/ChoroplethLegend";
import ChoroplethMap from "../components/ChoroplethMap";
import DashboardLeft from "../components/DashboardLeft";
import InfoCard from "../components/InfoCard";
import POILegend from "../components/POILegend";
import Plots from "../components/Plots";
import useFetchArticles from "../helpers/hooks/useFetchArticles";
import { Filters, InfoQuartiere, POI } from "../types/global";
import POIMarkers from "@/components/POIMarkers";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from "@/components/ui/collapsible";
import { format } from "date-fns";
import { GeoJsonObject } from "geojson";
import { LatLngExpression } from "leaflet";
import { ArrowUp, Loader2 } from "lucide-react";
import { BarChart3, ChevronsUpDown } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { MapContainer, TileLayer } from "react-leaflet";
import { useSearchParams } from "react-router-dom";

function Dashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [filters, setFilters] = useState<Filters>(() => {
    // Initialize from URL if present
    const urlStartDate = searchParams.get("startDate");
    const urlEndDate = searchParams.get("endDate");
    return {
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
        num_of_articles: 1
      },
      poi: {
        bar: 1,
        scommesse: 1,
        bancomat: 1,
        stazione: 1
      },
      subIndices: {
        poi: 1,
        socioEconomic: 1
      },
      dates: {
        startDate: urlStartDate ? new Date(urlStartDate) : null,
        endDate: urlEndDate ? new Date(urlEndDate) : null
      }
    };
  });
  const [tile, setTile] = useState<string>(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
  );
  const [palette, setPalette] = useState<string>("red");
  const [info, setInfo] = useState<InfoQuartiere>({
    name: "",
    crime_index: null,
    total_crimes: null,
    population: 0,
    crimes: []
  });
  const [data, setData] = useState<GeoJsonObject | null>(null);
  const [poi, setPoi] = useState<POI[]>([]);
  const [showPoi, setShowPoi] = useState<boolean>(true);
  const [legendValues, setLegendValues] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const { articles } = useFetchArticles(setIsLoading);

  const position: LatLngExpression = [41.117143, 16.871871];

  const handleResetDate = () => {
    setFilters((prev) => ({
      ...prev,
      dates: { startDate: null, endDate: null }
    }));
  };

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const queryParams = [];

      const selectedCrimes = Object.keys(filters.crimes)
        .filter((crime) => filters.crimes[crime] === 1)
        .join(",");
      queryParams.push(`crimes=${selectedCrimes}`);
      const selectedQuartieri = Object.keys(filters.quartieri)
        .filter((quartiere) => filters.quartieri[quartiere] === 1)
        .join(",");
      queryParams.push(`quartieri=${selectedQuartieri}`);

      const selectedPoi = Object.keys(filters.poi)
        .filter((poi) => filters.poi[poi] === 1)
        .join(",");
      queryParams.push(`poi=${selectedPoi}`);

      // Update URL with current filters
      const newSearchParams = new URLSearchParams();
      if (filters.dates.startDate) {
        const formattedStartDate = format(
          filters.dates.startDate,
          "yyyy-MM-dd"
        );
        queryParams.push(`startDate=${formattedStartDate}`);
        newSearchParams.set("startDate", formattedStartDate);
      }
      if (filters.dates.endDate) {
        const formattedEndDate = format(filters.dates.endDate, "yyyy-MM-dd");
        queryParams.push(`endDate=${formattedEndDate}`);
        newSearchParams.set("endDate", formattedEndDate);
      }
      setSearchParams(newSearchParams);

      queryParams.push(
        `${filters?.weights.num_of_articles === 1 ? "weightsForArticles=true" : "weightsForArticles=false"}`
      );

      // Sub-index toggles
      queryParams.push(
        `enablePoiSubIndex=${filters.subIndices.poi === 1 ? "true" : "false"}`
      );
      queryParams.push(
        `enableSocioEconomicSubIndex=${filters.subIndices.socioEconomic === 1 ? "true" : "false"}`
      );

      const queryString = queryParams.join("&");

      const response = await fetch(
        `http://127.0.0.1:3000/api/get-data?${queryString}`
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || `HTTP error! Status: ${response.status}`
        );
      }

      const jsonData = await response.json();
      setData(jsonData);
      // Extract POI from response
      setPoi(jsonData.poi || []);

      // Create legend
      const crimeIndexes: number[] = [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (jsonData as any)?.features.forEach((item: any) => {
        crimeIndexes.push(item.properties.crime_index);
      });
      // Fixed intervals: 0-20, 21-50, 51-80, 81-100
      setLegendValues([0, 21, 51, 81]);

      setInfo((prevState: InfoQuartiere) => ({
        ...prevState
      }));
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch data";
      setError(errorMessage);
      // eslint-disable-next-line no-console
      console.error("Error fetching dashboard data:", error);
    } finally {
      setIsLoading(false);
    }
  }, [filters, setSearchParams]);

  useEffect(() => {
    if (filters.dates.startDate === null) {
      setFilters((prev) => ({
        ...prev,
        dates: { ...prev.dates, endDate: null }
      }));
    }
  }, [filters.dates.startDate, setFilters]);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-8 xl:flex-row xl:gap-0">
      <div className="relative h-fit w-full p-4 xl:min-h-screen xl:w-[20%]">
        <DashboardLeft
          palette={palette}
          setPalette={setPalette}
          tile={tile}
          setTile={setTile}
          filters={filters}
          setFilters={setFilters}
          fetchData={fetchData}
          startDate={filters.dates.startDate}
          endDate={filters.dates.endDate}
          setEndDate={(value) => {
            const newDate =
              value instanceof Function ? value(filters.dates.endDate) : value;
            setFilters((prev) => ({
              ...prev,
              dates: { ...prev.dates, endDate: newDate }
            }));
          }}
          setStartDate={(value) => {
            const newDate =
              value instanceof Function
                ? value(filters.dates.startDate)
                : value;
            setFilters((prev) => ({
              ...prev,
              dates: { ...prev.dates, startDate: newDate }
            }));
          }}
          handleResetDate={handleResetDate}
          showPoi={showPoi}
          setShowPoi={setShowPoi}
        />
      </div>
      <div className="xl:border-border relative w-full px-4 xl:min-h-screen xl:w-[80%] xl:border-l xl:px-0">
        <div className="relative h-[800px] w-full bg-[#262626] xl:h-screen">
          <ArrowUp
            className="bg-primary text-primary-foreground hover:bg-primary/90 !sticky fixed !top-8 right-8 bottom-8 !left-2 !z-[10500] z-[10000] !h-12 h-12 !w-12 w-12 cursor-pointer rounded-full p-3 !text-white shadow-lg transition-all hover:shadow-xl xl:!hidden"
            onClick={scrollToTop}
          />
          <InfoCard
            name={info.name}
            crime_index={info.crime_index}
            crimes={info.crimes}
            population={info.population}
            poi_counts={info.poi_counts}
            sub_indices={info.sub_indices}
            activeSubIndices={filters.subIndices}
            showPoi={showPoi}
          />
          {error && (
            <div className="bg-destructive/10 border-destructive text-destructive absolute top-4 right-4 left-4 z-[1000] rounded-lg border p-4">
              <p className="font-medium">Error loading data</p>
              <p className="text-sm">{error}</p>
            </div>
          )}
          {isLoading ? (
            <Loader2 className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          ) : (
            <MapContainer
              className="h-full w-full"
              center={position}
              maxBoundsViscosity={1.0}
              zoom={12}
              scrollWheelZoom={true}>
              <TileLayer url={tile} />
              <ChoroplethMap
                setInfo={setInfo}
                data={data}
                color={palette}
                legendValues={legendValues}
              />
              {poi.length > 0 && showPoi && <POIMarkers poi={poi} />}
            </MapContainer>
          )}
          <ChoroplethLegend
            key={`legend-${legendValues.join("-")}`}
            palette={palette}
            legendValues={legendValues}
          />
          <POILegend visible={poi.length > 0 && showPoi} />
        </div>

        {data && (
          <Collapsible className="bg-card fixed right-0 bottom-0 z-[10000] w-full rounded-t-xl border-x border-t shadow-[0_-8px_30px_rgba(0,0,0,0.12)] xl:w-[80%]">
            <CollapsibleTrigger asChild={true}>
              <div className="group bg-primary text-primary-foreground hover:bg-primary/90 flex w-full cursor-pointer items-center justify-between rounded-t-xl px-6 py-4 transition-colors">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-5 w-5" />
                  <span className="text-lg font-semibold">View Analytics</span>
                </div>
                <ChevronsUpDown className="h-5 w-5 transition-transform duration-200 group-data-[state=open]:rotate-180" />
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="bg-card animate-slide-down data-[state=closed]:animate-slide-up max-h-[60vh] overflow-y-auto p-4">
              <Plots
                data={data}
                articles={articles}
                filters={filters}
                startDate={filters.dates.startDate}
                endDate={filters.dates.endDate}
              />
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}
export default Dashboard;
