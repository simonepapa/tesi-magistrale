import { getCrimeName } from "../helpers/utils";
import { Article, Filters } from "../types/global";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Feature, GeoJsonObject } from "geojson";
import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  XAxis,
  YAxis
} from "recharts";

type Props = {
  data: GeoJsonObject | null;
  articles: Article[] | null;
  filters: Filters;
  startDate: Date | null;
  endDate: Date | null;
};

// Crime-related properties to check
const crimeProperties = [
  "aggressione",
  "associazione_di_tipo_mafioso",
  "contrabbando",
  "estorsione",
  "furto",
  "omicidio",
  "omicidio_colposo",
  "omicidio_stradale",
  "rapina",
  "spaccio",
  "tentato_omicidio",
  "truffa",
  "violenza_sessuale"
];

const keyToLabels: { [key: string]: string } = {
  "bari-vecchia_san-nicola": "Bari Vecchia - San Nicola",
  carbonara: "Carbonara",
  carrassi: "Carrassi",
  "ceglie-del-campo": "Ceglie del Campo",
  japigia: "Japigia",
  liberta: "Libertà",
  loseto: "Loseto",
  madonnella: "Madonnella",
  murat: "Murat",
  "palese-macchie": "Palese - Macchie",
  picone: "Picone",
  "san-girolamo_fesca": "San Girolamo - Fesca",
  "san-paolo": "San Paolo",
  "san-pasquale": "San Pasquale",
  "santo-spirito": "Santo Spirito",
  stanic: "Stanic",
  "torre-a-mare": "Torre a mare"
};

function Plots({ data, articles, filters }: Props) {
  const [crimesByYear, setCrimesByYear] = useState<
    | {
        [key: string]: number | string;
      }[]
    | null
  >(null);
  const [crimesByYearQuartiere, setCrimesByYearQuartiere] = useState<
    | {
        [key: string]: number | string;
      }[]
    | null
  >(null);
  const [crimesByType, setCrimesByType] = useState<
    | {
        [key: string]: number | string;
      }[]
    | null
  >(null);
  const [hoveredLine, setHoveredLine] = useState<string | null>(null);

  const colors: string[] = [
    "#FF5733",
    "#33FF57",
    "#3357FF",
    "#FF33A1",
    "#A133FF",
    "#FFC300",
    "#FF5733",
    "#C70039",
    "#900C3F",
    "#581845",
    "#1F618D",
    "#28B463",
    "#D4AC0D",
    "#7D3C98",
    "#E74C3C",
    "#F39C12",
    "#2ECC71",
    "#3498DB",
    "#9B59B6",
    "#34495E"
  ];

  const barDataset = [
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ...(data as any).features.map((feature: Feature) => {
      return {
        ...feature.properties,
        name: feature?.properties?.name
      };
    })
  ];

  const countCrimesByYear = useCallback(() => {
    if (articles) {
      const crimeCountByYear: { [key: string]: number } = {};
      const articleCountByYear: { [key: string]: number } = {};
      const final: { [key: string]: number | string }[] = [];

      const filteredArticles = articles.filter((article) => {
        const quartiere = article.quartiere;
        // Check if quartiere is selected in filters
        // We need to map quartiere name to the key in filters.quartieri
        // The filters.quartieri keys are like "bari-vecchia_san-nicola"
        // The article.quartiere is like "bari-vecchia_san-nicola" (it comes from DB/API)
        // So we can check directly if it exists and is 1
        return (
          quartiere &&
          filters.quartieri[quartiere] === 1 &&
          (!filters.dates.startDate ||
            new Date(article.date) >= filters.dates.startDate) &&
          (!filters.dates.endDate ||
            new Date(article.date) <= filters.dates.endDate)
        );
      });

      filteredArticles.forEach((article) => {
        const year = new Date(article.date).getFullYear().toString();
        let crimes = 0;

        // Count crimes for this article
        const crimeSum = crimeProperties.reduce((sum, prop) => {
          return sum + (article[prop as keyof Article] === 1 ? 1 : 0);
        }, 0);
        crimes += crimeSum;

        crimeCountByYear[year] = (crimeCountByYear[year] || 0) + crimes;
        articleCountByYear[year] = (articleCountByYear[year] || 0) + 1;
      });

      Object.keys(crimeCountByYear).map((year: string) => {
        final.push({
          year: parseInt(year),
          label: year,
          crimes: crimeCountByYear[year],
          articles: articleCountByYear[year]
        });
      });

      setCrimesByYear(final);
    }
  }, [
    articles,
    filters.quartieri,
    filters.dates.startDate,
    filters.dates.endDate
  ]);

  const countCrimesByYearAndNeighborhood = useCallback(() => {
    if (articles) {
      const crimeData: { [key: string]: { [key: string]: number } } = {};

      const filteredArticles = articles.filter((article) => {
        const quartiere = article.quartiere;
        return (
          quartiere &&
          filters.quartieri[quartiere] === 1 &&
          (!filters.dates.startDate ||
            new Date(article.date) >= filters.dates.startDate) &&
          (!filters.dates.endDate ||
            new Date(article.date) <= filters.dates.endDate)
        );
      });

      filteredArticles.forEach((article) => {
        // quartiere is guaranteed to be defined and valid due to filter
        const quartiereSlug = article.quartiere!;
        const year = new Date(article.date).getFullYear().toString();
        let crimes = 0;

        const crimeSum = crimeProperties.reduce((sum, prop) => {
          return sum + (article[prop as keyof Article] === 1 ? 1 : 0);
        }, 0);
        crimes += crimeSum;

        if (!crimeData[quartiereSlug]) {
          crimeData[quartiereSlug] = {};
        }
        crimeData[quartiereSlug][year] =
          (crimeData[quartiereSlug][year] || 0) + crimes;
      });

      const allYears = new Set(
        Object.values(crimeData).flatMap((q) => Object.keys(q))
      );

      setCrimesByYearQuartiere(
        Array.from(allYears)
          .sort()
          .map((year) => {
            const entry: Record<string, number> = { year: parseInt(year) };
            for (const [quartiere, anni] of Object.entries(crimeData)) {
              entry[quartiere] = anni[year] || 0;
            }
            return entry;
          })
      );
    }
  }, [
    articles,
    filters.quartieri,
    filters.dates.startDate,
    filters.dates.endDate
  ]);

  const countCrimesByType = useCallback(() => {
    if (data) {
      let idCounter = 0;
      const crimeList: { [key: string]: number | string }[] = [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const filteredData = (data as any).features.filter((obj: any) => {
        return filters.quartieri[obj.properties.python_id] === 1;
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (filteredData as any).forEach((feature: Feature) => {
        if (feature && feature.properties && feature.properties.crimini) {
          Object.keys(feature?.properties?.crimini).map((key: string) => {
            const alreadyExists = crimeList.find(
              (crime: { [key: string]: number | string }) =>
                crime.label === getCrimeName(key)
            );
            if (alreadyExists) {
              alreadyExists.value +=
                feature?.properties?.crimini[key].frequenza;
            } else {
              crimeList.push({
                id: idCounter,
                value: feature?.properties?.crimini[key].frequenza,
                label: getCrimeName(key)
              });
              idCounter++;
            }
          });
        }
      });

      setCrimesByType(crimeList);
      return crimeList;
    }
  }, [data, filters.quartieri]);

  useEffect(() => {
    countCrimesByType();
    countCrimesByYear();
    countCrimesByYearAndNeighborhood();
  }, [countCrimesByType, countCrimesByYear, countCrimesByYearAndNeighborhood]);

  return (
    <div className="xl:pl-4">
      {data && (
        <Card className="bg-accent mt-4">
          <CardHeader>
            <CardTitle>Crime Risk Index breakdown per neighborhood</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs
              defaultValue={
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                (data as any).features.find(
                  (f: Feature) =>
                    filters.quartieri[f.properties?.python_id] === 1
                )?.properties?.python_id || ""
              }
              className="w-full">
              <TabsList className="mb-4 flex h-auto flex-wrap gap-1">
                {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                {(data as any).features
                  .filter(
                    (f: Feature) =>
                      filters.quartieri[f.properties?.python_id] === 1
                  )
                  .map((feature: Feature) => (
                    <TabsTrigger
                      key={feature.properties?.python_id}
                      value={feature.properties?.python_id}
                      className="hover:bg-primary/20 hover:text-primary data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-xs transition-all duration-200">
                      {keyToLabels[feature.properties?.python_id] ||
                        feature.properties?.name}
                    </TabsTrigger>
                  ))}
              </TabsList>

              {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
              {(data as any).features
                .filter(
                  (f: Feature) =>
                    filters.quartieri[f.properties?.python_id] === 1
                )
                .map((feature: Feature) => {
                  const props = feature.properties;
                  const subIndices = props?.sub_indices || {
                    S_crim: 0,
                    S_poi: 0,
                    S_soc: 0,
                    S_event: 0
                  };
                  const cri = props?.crime_index_scalato || 0;

                  const breakdownData = [
                    {
                      name: "CRI Breakdown",
                      Crime: subIndices.S_crim,
                      POI: filters.subIndices?.poi === 1 ? subIndices.S_poi : 0,
                      "Socio-Eco":
                        filters.subIndices?.socioEconomic === 1
                          ? subIndices.S_soc
                          : 0,
                      Event:
                        filters.subIndices?.event === 1 &&
                        subIndices.S_event > 0
                          ? subIndices.S_event
                          : 0
                    }
                  ];

                  return (
                    <TabsContent
                      key={feature.properties?.python_id}
                      value={feature.properties?.python_id}>
                      <div className="flex flex-col gap-4 xl:flex-row">
                        {/* Score Summary */}
                        <div className="flex flex-col gap-4 xl:w-1/3">
                          <div className="bg-background rounded-lg p-4">
                            <p className="text-muted-foreground text-sm">
                              Final Crime Risk Index
                            </p>
                            <p className="text-4xl font-bold">
                              {cri.toFixed(1)}
                            </p>
                            <p className="text-muted-foreground mt-1 text-xs">
                              Scale: 0-100 (higher = more risk, uses MinMax
                              scaling)
                            </p>
                          </div>

                          <div className="bg-background rounded-lg p-4">
                            <p className="text-muted-foreground mb-3 text-sm">
                              Sub-Index Values
                            </p>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="flex items-center gap-2">
                                  <div className="h-3 w-3 rounded bg-red-500" />
                                  Crime Index
                                </span>
                                <span className="font-semibold">
                                  {subIndices.S_crim.toFixed(1)}
                                </span>
                              </div>
                              {filters.subIndices?.poi === 1 && (
                                <div className="flex items-center justify-between">
                                  <span className="flex items-center gap-2">
                                    <div className="h-3 w-3 rounded bg-orange-500" />
                                    POI Index
                                  </span>
                                  <span className="font-semibold">
                                    {subIndices.S_poi.toFixed(1)}
                                  </span>
                                </div>
                              )}
                              {filters.subIndices?.socioEconomic === 1 && (
                                <div className="flex items-center justify-between">
                                  <span className="flex items-center gap-2">
                                    <div className="h-3 w-3 rounded bg-blue-500" />
                                    Socio-Economic Index
                                  </span>
                                  <span className="font-semibold">
                                    {subIndices.S_soc.toFixed(1)}
                                  </span>
                                </div>
                              )}
                              {filters.subIndices?.event === 1 &&
                                subIndices.S_event > 0 && (
                                  <div className="flex items-center justify-between">
                                    <span className="flex items-center gap-2">
                                      <div className="h-3 w-3 rounded bg-green-500" />
                                      Event Index
                                    </span>
                                    <span className="font-semibold">
                                      {subIndices.S_event.toFixed(1)}
                                    </span>
                                  </div>
                                )}
                            </div>
                          </div>

                          <div className="bg-background rounded-lg p-4">
                            <p className="text-muted-foreground mb-3 text-sm">
                              Risk Contribution
                            </p>
                            {(() => {
                              const sumOfActive =
                                subIndices.S_crim +
                                (filters.subIndices?.poi === 1
                                  ? subIndices.S_poi
                                  : 0) +
                                (filters.subIndices?.socioEconomic === 1
                                  ? subIndices.S_soc
                                  : 0) +
                                (filters.subIndices?.event === 1 &&
                                subIndices.S_event > 0
                                  ? subIndices.S_event
                                  : 0);
                              const crimeContrib =
                                sumOfActive > 0
                                  ? (subIndices.S_crim / sumOfActive) * 100
                                  : 0;
                              const poiContrib =
                                sumOfActive > 0 && filters.subIndices?.poi === 1
                                  ? (subIndices.S_poi / sumOfActive) * 100
                                  : 0;
                              const socContrib =
                                sumOfActive > 0 &&
                                filters.subIndices?.socioEconomic === 1
                                  ? (subIndices.S_soc / sumOfActive) * 100
                                  : 0;
                              const eventContrib =
                                sumOfActive > 0 &&
                                filters.subIndices?.event === 1 &&
                                subIndices.S_event > 0
                                  ? (subIndices.S_event / sumOfActive) * 100
                                  : 0;

                              return (
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="flex items-center gap-2">
                                      <div className="h-3 w-3 rounded bg-red-500" />
                                      Crime
                                    </span>
                                    <span className="font-semibold">
                                      {crimeContrib.toFixed(1)}%
                                    </span>
                                  </div>
                                  {filters.subIndices?.poi === 1 && (
                                    <div className="flex items-center justify-between">
                                      <span className="flex items-center gap-2">
                                        <div className="h-3 w-3 rounded bg-orange-500" />
                                        POI
                                      </span>
                                      <span className="font-semibold">
                                        {poiContrib.toFixed(1)}%
                                      </span>
                                    </div>
                                  )}
                                  {filters.subIndices?.socioEconomic === 1 && (
                                    <div className="flex items-center justify-between">
                                      <span className="flex items-center gap-2">
                                        <div className="h-3 w-3 rounded bg-blue-500" />
                                        Socio-Eco
                                      </span>
                                      <span className="font-semibold">
                                        {socContrib.toFixed(1)}%
                                      </span>
                                    </div>
                                  )}
                                  {filters.subIndices?.event === 1 &&
                                    subIndices.S_event > 0 && (
                                      <div className="flex items-center justify-between">
                                        <span className="flex items-center gap-2">
                                          <div className="h-3 w-3 rounded bg-green-500" />
                                          Event
                                        </span>
                                        <span className="font-semibold">
                                          {eventContrib.toFixed(1)}%
                                        </span>
                                      </div>
                                    )}
                                </div>
                              );
                            })()}
                          </div>
                        </div>

                        <div className="xl:w-2/3">
                          <ChartContainer
                            config={
                              {
                                Crime: {
                                  label: "Crime Index",
                                  color: "#ef4444"
                                },
                                POI: {
                                  label: "POI Index",
                                  color: "#f97316"
                                },
                                "Socio-Eco": {
                                  label: "Socio-Economic Index",
                                  color: "#3b82f6"
                                },
                                Event: {
                                  label: "Event Index",
                                  color: "#22c55e"
                                }
                              } satisfies ChartConfig
                            }
                            className="min-h-[200px] w-full">
                            <ResponsiveContainer width="100%" height={200}>
                              <BarChart
                                data={breakdownData}
                                layout="vertical"
                                margin={{
                                  left: 20,
                                  right: 20,
                                  top: 20,
                                  bottom: 20
                                }}>
                                <CartesianGrid
                                  strokeDasharray="3 3"
                                  horizontal={false}
                                />
                                <XAxis type="number" domain={[0, 100]} />
                                <YAxis
                                  type="category"
                                  dataKey="name"
                                  hide={true}
                                />
                                <ChartTooltip
                                  content={<ChartTooltipContent />}
                                />
                                <Legend />
                                <Bar
                                  dataKey="Crime"
                                  stackId="a"
                                  fill="#ef4444"
                                  radius={[4, 0, 0, 4]}
                                />
                                {filters.subIndices?.poi === 1 && (
                                  <Bar
                                    dataKey="POI"
                                    stackId="a"
                                    fill="#f97316"
                                  />
                                )}
                                {filters.subIndices?.socioEconomic === 1 && (
                                  <Bar
                                    dataKey="Socio-Eco"
                                    stackId="a"
                                    fill="#3b82f6"
                                    radius={[0, 4, 4, 0]}
                                  />
                                )}
                                {filters.subIndices?.event === 1 &&
                                  subIndices.S_event > 0 && (
                                    <Bar
                                      dataKey="Event"
                                      stackId="a"
                                      fill="#22c55e"
                                      radius={[0, 4, 4, 0]}
                                    />
                                  )}
                              </BarChart>
                            </ResponsiveContainer>
                          </ChartContainer>
                        </div>
                      </div>
                    </TabsContent>
                  );
                })}
            </Tabs>
          </CardContent>
        </Card>
      )}

      {data && barDataset && crimesByType && (
        <div className="mt-4 flex flex-col gap-4 xl:flex-row">
          <Card className="bg-accent w-full xl:w-1/2">
            <CardHeader>
              <CardTitle>Crime index per neighborhood</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={
                  {
                    crimeIndex: {
                      label: "Standardized Crime Index (0-100)",
                      color: "var(--primary)"
                    }
                  } satisfies ChartConfig
                }
                className="min-h-[400px] w-full">
                <BarChart
                  accessibilityLayer={true}
                  data={barDataset}
                  margin={{ bottom: 80, left: 12, right: 12 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="name"
                    tickLine={false}
                    tickMargin={10}
                    axisLine={false}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tickLine={false} axisLine={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="crime_index" fill="var(--primary)" radius={4} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
          <Card className="bg-accent w-full xl:w-1/2">
            <CardHeader>
              <CardTitle>Most common crimes in Bari</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={
                  crimesByType?.reduce((acc, crime, index) => {
                    acc[crime.label as string] = {
                      label: crime.label as string,
                      color: colors[index % colors.length]
                    };
                    return acc;
                  }, {} as ChartConfig) || {}
                }
                className="min-h-[400px] w-full">
                <PieChart>
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />
                  <Pie
                    data={crimesByType || []}
                    dataKey="value"
                    nameKey="label"
                    cx="50%"
                    cy="50%"
                    outerRadius={120}>
                    {crimesByType?.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={colors[index % colors.length]}
                      />
                    ))}
                  </Pie>
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>
      )}
      {articles && (
        <Card className="bg-accent mt-4">
          <CardHeader>
            <CardTitle>Evolution of crimes in Bari by year</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-4 xl:flex-row">
              <Card className="w-full xl:flex-1">
                <CardHeader>
                  <CardTitle className="text-base">
                    Crimes and Articles by Year
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer
                    config={
                      {
                        crimes: {
                          label: "Crime Articles",
                          color: "var(--primary)"
                        },
                        articles: {
                          label: "Total Articles",
                          color: "var(--chart-2)"
                        }
                      } satisfies ChartConfig
                    }
                    className="min-h-[400px] w-full">
                    <LineChart
                      data={crimesByYear || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="year" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend
                        onMouseEnter={(e) =>
                          setHoveredLine(e.dataKey as string)
                        }
                        onMouseLeave={() => setHoveredLine(null)}
                      />
                      <Line
                        type="monotone"
                        dataKey="crimes"
                        stroke="var(--primary)"
                        strokeWidth={hoveredLine === "crimes" ? 6 : 4}
                        strokeOpacity={
                          hoveredLine && hoveredLine !== "crimes" ? 0.2 : 1
                        }
                        dot={{ fill: "var(--primary)", r: 4 }}
                        onMouseEnter={() => setHoveredLine("crimes")}
                        onMouseLeave={() => setHoveredLine(null)}
                      />
                      <Line
                        type="monotone"
                        dataKey="articles"
                        stroke="var(--chart-2)"
                        strokeWidth={hoveredLine === "articles" ? 6 : 4}
                        strokeOpacity={
                          hoveredLine && hoveredLine !== "articles" ? 0.2 : 1
                        }
                        dot={{ fill: "var(--chart-2)", r: 4 }}
                        onMouseEnter={() => setHoveredLine("articles")}
                        onMouseLeave={() => setHoveredLine(null)}
                      />
                    </LineChart>
                  </ChartContainer>
                </CardContent>
              </Card>
              <Card className="w-full xl:flex-1">
                <CardHeader>
                  <CardTitle className="text-base">
                    Crimes by Year and neighborhood
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer
                    config={Object.keys(keyToLabels).reduce(
                      (acc, key, index) => {
                        acc[key] = {
                          label: keyToLabels[key],
                          color: colors[index % colors.length]
                        };
                        return acc;
                      },
                      {} as ChartConfig
                    )}
                    className="min-h-[250px] w-full">
                    <LineChart
                      data={crimesByYearQuartiere || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="year" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend
                        onMouseEnter={(e) =>
                          setHoveredLine(e.dataKey as string)
                        }
                        onMouseLeave={() => setHoveredLine(null)}
                      />
                      {Object.keys(keyToLabels)
                        .filter((key) => filters.quartieri[key] === 1)
                        .map((key) => {
                          const index = Object.keys(keyToLabels).indexOf(key);
                          return (
                            <Line
                              key={key}
                              type="monotone"
                              dataKey={key}
                              stroke={colors[index % colors.length]}
                              strokeWidth={hoveredLine === key ? 5 : 3}
                              strokeOpacity={
                                hoveredLine && hoveredLine !== key ? 0.2 : 1
                              }
                              dot={false}
                              onMouseEnter={() => setHoveredLine(key)}
                              onMouseLeave={() => setHoveredLine(null)}
                            />
                          );
                        })}
                    </LineChart>
                  </ChartContainer>
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
export default Plots;
