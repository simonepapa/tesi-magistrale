import { getCrimeName, getQuartiereName } from "../helpers/utils";
import { Article } from "../types/global";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DateRangePicker } from "@/components/ui/date-range-picker";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { format, startOfYear, endOfYear } from "date-fns";
import { Loader2, ArrowUp, ArrowDown, ChevronsUpDown } from "lucide-react";
import { useState, useEffect } from "react";
import { DateRange } from "react-day-picker";
import { useSearchParams } from "react-router-dom";

const neighborhoods = [
  { value: "bari-vecchia_san-nicola", label: "Bari Vecchia - San Nicola" },
  { value: "carbonara", label: "Carbonara" },
  { value: "carrassi", label: "Carrassi" },
  { value: "ceglie-del-campo", label: "Ceglie del Campo" },
  { value: "japigia", label: "Japigia" },
  { value: "liberta", label: "Libertà" },
  { value: "loseto", label: "Loseto" },
  { value: "madonnella", label: "Madonnella" },
  { value: "murat", label: "Murat" },
  { value: "palese-macchie", label: "Palese - Macchie" },
  { value: "picone", label: "Picone" },
  { value: "san-girolamo_fesca", label: "San Girolamo - Fesca" },
  { value: "san-paolo", label: "San Paolo" },
  { value: "san-pasquale", label: "San Pasquale" },
  { value: "santo-spirito", label: "Santo Spirito" },
  { value: "stanic", label: "Stanic" },
  { value: "torre-a-mare", label: "Torre a mare" }
];

const crimeCategories = [
  { key: "aggressione", label: getCrimeName("aggressione") },
  {
    key: "associazione_di_tipo_mafioso",
    label: getCrimeName("associazione_di_tipo_mafioso")
  },
  { key: "contrabbando", label: getCrimeName("contrabbando") },
  { key: "estorsione", label: getCrimeName("estorsione") },
  { key: "furto", label: getCrimeName("furto") },
  { key: "omicidio", label: getCrimeName("omicidio") },
  { key: "omicidio_colposo", label: getCrimeName("omicidio_colposo") },
  { key: "omicidio_stradale", label: getCrimeName("omicidio_stradale") },
  { key: "rapina", label: getCrimeName("rapina") },
  { key: "spaccio", label: getCrimeName("spaccio") },
  { key: "tentato_omicidio", label: getCrimeName("tentato_omicidio") },
  { key: "truffa", label: getCrimeName("truffa") },
  { key: "violenza_sessuale", label: getCrimeName("violenza_sessuale") }
];

function ReadArticles() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [articles, setArticles] = useState<Article[]>([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  // Filter states initialized from URL
  const [selectedQuartiere, setSelectedQuartiere] = useState<
    string | undefined
  >(searchParams.get("quartiere") || undefined);

  const [dateRange, setDateRange] = useState<DateRange | undefined>(() => {
    const start = searchParams.get("start");
    const end = searchParams.get("end");
    return {
      from: start ? new Date(start) : startOfYear(new Date()),
      to: end ? new Date(end) : endOfYear(new Date())
    };
  });

  const [page, setPage] = useState<number>(() => {
    const pageParam = searchParams.get("page");
    return pageParam ? parseInt(pageParam) : 1;
  });
  const [limit, setLimit] = useState<number>(() => {
    const limitParam = searchParams.get("limit");
    return limitParam ? parseInt(limitParam) : 10;
  });
  const [totalPages, setTotalPages] = useState<number>(1);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (newPage: number = 1, newLimit?: number) => {
    const currentLimit = newLimit || limit;
    setIsLoading(true);
    setHasSearched(true);
    setSelectedArticle(null);

    try {
      const params = new URLSearchParams();

      if (selectedQuartiere) {
        params.append("quartiere", selectedQuartiere);
      }

      if (dateRange?.from && dateRange?.to) {
        const startStr = format(dateRange.from, "yyyy-MM-dd");
        const endStr = format(dateRange.to, "yyyy-MM-dd");
        params.append("startDate", startStr);
        params.append("endDate", endStr);
      }

      params.append("page", newPage.toString());
      params.append("limit", currentLimit.toString());

      // Update URL params
      const newSearchParams = new URLSearchParams(searchParams);
      if (selectedQuartiere) {
        newSearchParams.set("quartiere", selectedQuartiere);
      } else {
        newSearchParams.delete("quartiere");
      }
      if (dateRange?.from && dateRange?.to) {
        newSearchParams.set("start", format(dateRange.from, "yyyy-MM-dd"));
        newSearchParams.set("end", format(dateRange.to, "yyyy-MM-dd"));
      }
      newSearchParams.set("page", newPage.toString());
      newSearchParams.set("limit", currentLimit.toString());
      setSearchParams(newSearchParams);
      setPage(newPage);
      if (newLimit) setLimit(newLimit);

      const url = `http://127.0.0.1:3000/api/get-articles?${params.toString()}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const jsonData = await response.json();
      if (jsonData.articles) {
        setArticles(jsonData.articles);
        setTotalPages(jsonData.totalPages);
      } else {
        // Fallback if backend returns array (shouldn't happen with page param)
        setArticles(jsonData);
        setTotalPages(1);
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Request error", error);
      setArticles([]);
    }

    setIsLoading(false);
  };

  // Trigger search on mount if params exist
  useEffect(() => {
    if (
      searchParams.has("start") ||
      searchParams.has("quartiere") ||
      searchParams.has("page") ||
      searchParams.has("limit")
    ) {
      const pageParam = searchParams.get("page");
      const limitParam = searchParams.get("limit");
      handleSearch(
        pageParam ? parseInt(pageParam) : 1,
        limitParam ? parseInt(limitParam) : 10
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const downloadJSON = () => {
    if (!articles || articles.length === 0) return;

    const updatedJson = JSON.stringify(articles, null, 2);
    const blob = new Blob([updatedJson], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download =
      "articles_filtered-" +
      format(new Date(), "yyyy-MM-dd_HH-mm-ss") +
      ".json";
    link.click();
  };

  const handleCardClick = (article: Article) => {
    setSelectedArticle(article);
    setIsDialogOpen(true);
  };

  const handleDialogClose = () => {
    setIsDialogOpen(false);
    setTimeout(() => setSelectedArticle(null), 200);
  };

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  const scrollToBottom = () => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "smooth"
    });
  };

  const handleReset = () => {
    setSelectedQuartiere(undefined);
    setDateRange({
      from: startOfYear(new Date()),
      to: endOfYear(new Date())
    });
    setHasSearched(false);
    setArticles([]);
    setPage(1);
    setTotalPages(1);
    setLimit(10);
    setSearchParams(new URLSearchParams());
  };

  return (
    <>
      <ArrowUp
        className="bg-primary text-primary-foreground hover:bg-primary/90 fixed right-8 bottom-8 z-[10000] h-12 w-12 cursor-pointer rounded-full p-3 shadow-lg transition-all hover:shadow-xl"
        onClick={scrollToTop}
      />
      <ArrowDown
        className="bg-primary text-primary-foreground hover:bg-primary/90 fixed right-24 bottom-8 z-[10000] h-12 w-12 cursor-pointer rounded-full p-3 shadow-lg transition-all hover:shadow-xl"
        onClick={scrollToBottom}
      />
      <div className="mt-8 mb-12 flex flex-col gap-8 px-4 lg:mx-auto lg:max-w-[1400px] xl:flex-row xl:px-0">
        <div className="w-full xl:sticky xl:top-8 xl:w-[30%] xl:self-start">
          <h1 className="mb-2 text-3xl font-bold">Read Articles</h1>
          <p className="text-muted-foreground mb-6">
            Filter articles by neighborhood and date range
          </p>

          <Card className="gap-2">
            <CardHeader>
              <CardTitle>Filter news</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col gap-1">
                <label className="text-sm font-medium">Neighborhood</label>
                <Select
                  value={selectedQuartiere || "all"}
                  onValueChange={(value) =>
                    setSelectedQuartiere(value === "all" ? undefined : value)
                  }>
                  <SelectTrigger>
                    <SelectValue placeholder="All neighborhoods" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All neighborhoods</SelectItem>
                    {neighborhoods.map((neighborhood) => (
                      <SelectItem
                        key={neighborhood.value}
                        value={neighborhood.value}>
                        {neighborhood.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-sm font-medium">Date Range</label>
                <DateRangePicker value={dateRange} onChange={setDateRange} />
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={() => handleSearch(1)}
                  className="h-10 flex-1"
                  disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    "Search Articles"
                  )}
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  Reset
                </Button>
              </div>

              {articles && articles.length > 0 && (
                <Button
                  onClick={downloadJSON}
                  variant="outline"
                  className="w-full">
                  Download JSON ({articles.length} articles)
                </Button>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="w-full xl:w-[70%]">
          {hasSearched && (
            <>
              <div className="mb-6">
                <h2 className="mb-2 text-2xl font-semibold">
                  {articles
                    ? `${articles.length} ${articles.length === 1 ? "Article" : "Articles"} Found`
                    : "No articles"}
                </h2>
                <p className="text-muted-foreground text-sm">
                  Note that <span className="italic">xx% of being true</span>{" "}
                  means that the category is <span className="italic">xx%</span>{" "}
                  likely to be true and has nothing to do with the prediction
                  being correct. The threshold for a category to be true is set
                  to 75%.
                </p>
              </div>

              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="text-primary h-8 w-8 animate-spin" />
                </div>
              ) : articles && articles.length > 0 ? (
                <div className="grid grid-cols-1 gap-4">
                  {articles.map((article) => (
                    <Card
                      key={article.id}
                      className="hover:bg-card/80 cursor-pointer transition-all hover:shadow-lg"
                      onClick={() => handleCardClick(article)}>
                      <CardHeader>
                        <div className="flex items-start justify-between gap-4">
                          <CardTitle className="text-lg">
                            {article.title}
                          </CardTitle>
                          <ChevronsUpDown className="text-muted-foreground h-5 w-5 shrink-0" />
                        </div>
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                          <Badge variant="secondary" className="font-normal">
                            {getQuartiereName(article.quartiere || "")}
                          </Badge>
                          <p className="text-muted-foreground text-sm">
                            Published on:{" "}
                            {format(new Date(article.date), "MMMM d, yyyy")}
                          </p>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <p className="text-muted-foreground line-clamp-2 text-sm">
                          {article.content}
                        </p>

                        <div className="flex flex-wrap gap-2">
                          {crimeCategories.map(({ key, label }) => {
                            const isTrue =
                              (article as unknown as Record<string, number>)[
                                key
                              ] === 1;
                            const probability = (
                              ((article as unknown as Record<string, number>)[
                                `${key}_prob`
                              ] || 0) * 100
                            ).toFixed(0);

                            return (
                              <Badge
                                key={key}
                                variant={isTrue ? "default" : "outline"}
                                className={`text-xs ${
                                  isTrue
                                    ? "border-green-600 bg-green-600 text-white hover:bg-green-700"
                                    : ""
                                }`}>
                                {label}: {isTrue ? "true" : "false"} (
                                {probability}%)
                              </Badge>
                            );
                          })}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center">
                  No articles found for the selected filters.
                </p>
              )}

              {articles && articles.length > 0 && (
                <div className="mt-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground text-sm">
                      Rows per page
                    </span>
                    <Select
                      value={limit.toString()}
                      onValueChange={(value) => {
                        const newLimit = parseInt(value);
                        setLimit(newLimit);
                        if (hasSearched) {
                          handleSearch(1, newLimit);
                        }
                      }}>
                      <SelectTrigger className="h-8 w-[70px]">
                        <SelectValue placeholder="10" />
                      </SelectTrigger>
                      <SelectContent side="top">
                        <SelectItem value="10">10</SelectItem>
                        <SelectItem value="20">20</SelectItem>
                        <SelectItem value="50">50</SelectItem>
                        <SelectItem value="100">100</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center gap-4">
                    <Button
                      variant="outline"
                      onClick={() => {
                        const newPage = Math.max(1, page - 1);
                        handleSearch(newPage);
                        scrollToTop();
                      }}
                      disabled={page === 1 || isLoading}>
                      Previous
                    </Button>
                    <span className="text-sm font-medium">
                      Page {page} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const newPage = Math.min(totalPages, page + 1);
                        handleSearch(newPage);
                        scrollToTop();
                      }}
                      disabled={page === totalPages || isLoading}>
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}

          {!hasSearched && (
            <div className="text-muted-foreground flex h-[400px] items-center justify-center">
              <p>Use the filters on the left to search for articles</p>
            </div>
          )}
        </div>

        <Dialog open={isDialogOpen} onOpenChange={handleDialogClose}>
          <DialogContent className="bg-card max-h-[85vh] w-[95%] max-w-3xl overflow-y-auto rounded-lg">
            {selectedArticle && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-2xl">
                    {selectedArticle.title}
                  </DialogTitle>
                  <div className="text-muted-foreground flex items-center gap-2 pt-2 text-sm">
                    <Badge variant="secondary" className="font-normal">
                      {getQuartiereName(selectedArticle.quartiere || "")}
                    </Badge>
                    <span>
                      Published on:{" "}
                      {format(new Date(selectedArticle.date), "MMMM d, yyyy")}
                    </span>
                  </div>
                </DialogHeader>

                <div className="space-y-6 pt-4">
                  <div className="prose max-w-none">
                    {selectedArticle.content
                      .split("\n")
                      .map((str: string, index: number) => (
                        <p key={index} className="mb-2 text-sm">
                          {str}
                        </p>
                      ))}
                  </div>

                  <div className="border-t pt-4">
                    <h3 className="mb-3 text-lg font-semibold">
                      Crime Categories
                    </h3>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                      {crimeCategories.map(({ key, label }) => {
                        const isTrue =
                          (
                            selectedArticle as unknown as Record<string, number>
                          )[key] === 1;
                        const probability = (
                          ((
                            selectedArticle as unknown as Record<string, number>
                          )[`${key}_prob`] || 0) * 100
                        ).toFixed(2);

                        return (
                          <div
                            key={key}
                            className="flex items-center justify-between">
                            <span className="text-sm font-medium">
                              {label}:
                            </span>
                            <Badge
                              variant={isTrue ? "default" : "outline"}
                              className={
                                isTrue
                                  ? "border-green-600 bg-green-600 text-white hover:bg-green-700"
                                  : ""
                              }>
                              {isTrue ? "True" : "False"} ({probability}%)
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </>
  );
}

export default ReadArticles;
