import { getCrimeName } from "../helpers/utils";
import { Article, LabeledArticle } from "../types/global";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import {
  ArrowLeft,
  ArrowRight,
  Database,
  Upload,
  Loader2,
  Sparkles,
  AlertCircle,
  RotateCcw,
  Check,
  Menu
} from "lucide-react";
import { enqueueSnackbar } from "notistack";
import { ChangeEvent, SyntheticEvent, useCallback, useState } from "react";

function LabelArticles() {
  const [articles, setArticles] = useState<Article[] | null>(null);
  const [labeledArticles, setLabeledArticles] = useState<
    LabeledArticle[] | null
  >(null);
  const [currentArticle, setCurrentArticle] = useState<number>(0);
  const [quartiere, setQuartiere] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [reviewedArticles, setReviewedArticles] = useState<Set<number>>(
    new Set()
  );
  const [filter, setFilter] = useState<"all" | "to_review" | "reviewed">("all");

  const handleChangeQuartiere = (value: string) => {
    if (!isLoading) {
      setQuartiere(value);
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!isLoading) {
      if (!event.target.files) {
        setError("Please upload a valid JSON file.");
      } else {
        setError("");

        const file = event.target.files[0];
        if (file && file.type === "application/json") {
          const reader = new FileReader();
          reader.onload = () => {
            try {
              if (!reader.result) {
                setError("Error reading the file");
              } else {
                const data: Article[] = (JSON.parse(reader.result as string) ||
                  "[{}]") as Article[];

                setArticles(data);
                setLabeledArticles(null); // Clear previous results
                setCurrentArticle(0); // Reset pagination
              }
            } catch (error) {
              setError("Failed to parse JSON: " + error);
            }
          };
          reader.readAsText(file);
        } else {
          setError("Please upload a valid JSON file.");
        }

        // Reset input value to allow selecting the same file again
        event.target.value = "";
      }
    }
  };

  const handleSubmit = useCallback(
    async (e: SyntheticEvent) => {
      e.preventDefault();

      if (!isLoading) {
        setIsLoading(true);

        try {
          const response = await fetch(
            `http://127.0.0.1:5000/classifier/label-articles`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                jsonFile: articles,
                quartiere: quartiere
              })
            }
          );

          if (response.ok) {
            const labeledArticles = await response.json();
            setLabeledArticles(labeledArticles);
          } else {
            enqueueSnackbar(`Response error: ${response.status}`, {
              variant: "error"
            });
          }
        } catch (error) {
          enqueueSnackbar(`Request error: ${error}`, { variant: "error" });
        }

        setIsLoading(false);
      }
    },
    [articles, isLoading, quartiere]
  );

  const handlePrevArticle = () => {
    if (currentArticle > 0) {
      setCurrentArticle(currentArticle - 1);
    }
  };
  const handleNextArticle = () => {
    if (labeledArticles && currentArticle < labeledArticles.length - 1) {
      setReviewedArticles((prev) => new Set(prev).add(currentArticle));
      setCurrentArticle(currentArticle + 1);
    }
  };
  const handleUploadToDatabase = useCallback(async () => {
    if (!isLoading && !isUploading) {
      setIsUploading(true);

      try {
        const response = await fetch(
          `http://127.0.0.1:3000/api/upload-to-database`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              jsonFile: labeledArticles
            })
          }
        );

        if (response.ok) {
          setArticles(null);
          setLabeledArticles(null);
          setCurrentArticle(0);
          setQuartiere("");
          enqueueSnackbar("Articles uploaded succesfully", {
            variant: "success"
          });
        } else {
          enqueueSnackbar(
            "Something went wrong while uploading articles to database",
            {
              variant: "error"
            }
          );
          enqueueSnackbar(`Response error: ${response.status}`, {
            variant: "error"
          });
        }
      } catch (error) {
        enqueueSnackbar(`Request error: ${error}`, { variant: "error" });
      }

      setIsUploading(false);
    }
  }, [isLoading, isUploading, labeledArticles]);

  const handleChangeLabel = (checked: boolean, cat: string) => {
    const categories = [
      "omicidio",
      "omicidio_colposo",
      "omicidio_stradale",
      "tentato_omicidio",
      "furto",
      "rapina",
      "violenza_sessuale",
      "aggressione",
      "spaccio",
      "truffa",
      "estorsione",
      "contrabbando",
      "associazione_di_tipo_mafioso"
    ];

    if (labeledArticles && categories.includes(cat)) {
      const copyLabeled = [...labeledArticles];
      // const value = copyLabeled[currentArticle][cat].value;
      copyLabeled[currentArticle][cat].value = checked ? 1 : 0;

      setLabeledArticles(copyLabeled);
    }
  };

  const handleReset = useCallback(() => {
    setArticles(null);
    setLabeledArticles(null);
    setCurrentArticle(0);
    setQuartiere("");
    setError("");
    setIsLoading(false);
    setIsUploading(false);
    setReviewedArticles(new Set());
    setFilter("all");
  }, []);

  const handleArticleSelect = (index: number) => {
    setCurrentArticle(index);
  };

  const filteredArticles = labeledArticles
    ?.map((article, index) => ({ ...article, originalIndex: index }))
    .filter((article) => {
      if (filter === "all") return true;
      if (filter === "to_review")
        return !reviewedArticles.has(article.originalIndex);
      if (filter === "reviewed")
        return reviewedArticles.has(article.originalIndex);
      return true;
    });

  const categories = [
    "omicidio",
    "omicidio_colposo",
    "omicidio_stradale",
    "tentato_omicidio",
    "furto",
    "rapina",
    "violenza_sessuale",
    "aggressione",
    "spaccio",
    "truffa",
    "estorsione",
    "contrabbando",
    "associazione_di_tipo_mafioso"
  ];

  return (
    <>
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Label Articles</h1>
          <p className="text-muted-foreground mt-2">
            Automatically label articles using a BERT model, then revise them
            and upload them to the database
          </p>
          <Alert className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="mt-1">
              <span className="font-medium">IMPORTANT:</span> This function
              works best with a dedicated GPU. CPU labeling is possible but
              requires significantly more time.
            </AlertDescription>
          </Alert>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Upload & Analyze</CardTitle>
              <CardDescription>
                Follow these steps to label your articles
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Badge variant="default">Step 1</Badge>
                  Upload JSON File
                </Label>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  disabled={isLoading}
                  asChild={true}>
                  <label className="cursor-pointer">
                    <Upload className="mr-2 h-4 w-4" />
                    {articles
                      ? `${articles.length} articles loaded`
                      : "Choose File"}
                    <input
                      type="file"
                      className="hidden"
                      onChange={handleFileChange}
                      accept="application/json"
                    />
                  </label>
                </Button>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Badge variant="default">Step 2</Badge>
                  Select Neighborhood
                </Label>
                <Select
                  value={quartiere}
                  onValueChange={handleChangeQuartiere}
                  disabled={isLoading}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a neighborhood..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bari-vecchia_san-nicola">
                      Bari Vecchia - San Nicola
                    </SelectItem>
                    <SelectItem value="carbonara">Carbonara</SelectItem>
                    <SelectItem value="carrassi">Carrassi</SelectItem>
                    <SelectItem value="ceglie-del-campo">
                      Ceglie del Campo
                    </SelectItem>
                    <SelectItem value="japigia">Japigia</SelectItem>
                    <SelectItem value="liberta">Libertà</SelectItem>
                    <SelectItem value="loseto">Loseto</SelectItem>
                    <SelectItem value="madonnella">Madonnella</SelectItem>
                    <SelectItem value="murat">Murat</SelectItem>
                    <SelectItem value="palese-macchie">
                      Palese - Macchie
                    </SelectItem>
                    <SelectItem value="picone">Picone</SelectItem>
                    <SelectItem value="san-girolamo_fesca">
                      San Girolamo - Fesca
                    </SelectItem>
                    <SelectItem value="san-paolo">San Paolo</SelectItem>
                    <SelectItem value="san-pasquale">San Pasquale</SelectItem>
                    <SelectItem value="santo-spirito">Santo Spirito</SelectItem>
                    <SelectItem value="stanic">Stanic</SelectItem>
                    <SelectItem value="torre-a-mare">Torre a mare</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <Button
                className="w-full"
                disabled={!articles || !quartiere || isLoading}
                onClick={(e) => {
                  e.preventDefault();
                  handleSubmit(e);
                }}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Start Analysis
                  </>
                )}
              </Button>

              <Button
                variant="outline"
                className="w-full"
                disabled={isLoading}
                onClick={(e) => {
                  e.preventDefault();
                  handleReset();
                }}>
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset
              </Button>

              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card className={articles ? "hidden lg:block" : ""}>
            <CardHeader>
              <CardTitle>Expected JSON Format</CardTitle>
              <CardDescription>
                Upload a JSON array with objects in this format
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted overflow-x-auto rounded-lg p-4 text-sm">
                <code>{`{
    "link": "https://example.com/article",
    "title": "Article Title",
    "date": "YYYY-MM-DD HH:mm:ss",
    "content": "Article content here..."
},
{
    "link": "https://example.com/article2",
    "title": "Article Title 2",
    "date": "YYYY-MM-DD HH:mm:ss",
    "content": "Article2 content here..."
}`}</code>
              </pre>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="container mx-auto px-4 pb-8">
        {labeledArticles && (
          <>
            {/* Mobile Article List Trigger */}
            <div className="fixed bottom-6 left-6 z-50 lg:hidden">
              <Sheet>
                <SheetTrigger asChild={true}>
                  <Button
                    size="icon"
                    className="h-12 w-12 rounded-full shadow-lg">
                    <Menu className="h-6 w-6" />
                  </Button>
                </SheetTrigger>
                <SheetContent
                  side="left"
                  className="w-[300px] p-0 sm:w-[400px]">
                  <div className="h-full p-4 pt-10">
                    <div className="flex h-full flex-col gap-4">
                      <div className="flex flex-wrap gap-2">
                        <Badge
                          variant={filter === "all" ? "default" : "outline"}
                          className="cursor-pointer px-4 py-2 text-sm transition-colors"
                          onClick={() => setFilter("all")}>
                          All
                        </Badge>
                        <Badge
                          variant={
                            filter === "to_review" ? "default" : "outline"
                          }
                          className="cursor-pointer px-4 py-2 text-sm transition-colors"
                          onClick={() => setFilter("to_review")}>
                          To Review
                        </Badge>
                        <Badge
                          variant={
                            filter === "reviewed" ? "default" : "outline"
                          }
                          className="cursor-pointer px-4 py-2 text-sm transition-colors"
                          onClick={() => setFilter("reviewed")}>
                          Reviewed
                        </Badge>
                      </div>

                      <Card className="flex flex-1 flex-col overflow-hidden">
                        <CardHeader className="border-b px-4 py-3">
                          <CardTitle className="flex items-center justify-between text-sm font-medium">
                            <span>Articles List</span>
                            <Badge variant="secondary">
                              {filteredArticles?.length} /{" "}
                              {labeledArticles.length}
                            </Badge>
                          </CardTitle>
                        </CardHeader>
                        <div className="flex-1 space-y-2 overflow-y-auto p-2">
                          {filteredArticles?.map((article) => (
                            <div
                              key={article.originalIndex}
                              onClick={() =>
                                handleArticleSelect(article.originalIndex)
                              }
                              className={`hover:bg-accent cursor-pointer rounded-lg border p-3 transition-colors ${
                                currentArticle === article.originalIndex
                                  ? "bg-accent border-primary"
                                  : "bg-card"
                              }`}>
                              <div className="mb-1 flex items-start justify-between">
                                <h4 className="line-clamp-1 text-sm font-medium">
                                  {article.title}
                                </h4>
                                {reviewedArticles.has(
                                  article.originalIndex
                                ) && (
                                  <Check className="h-4 w-4 flex-shrink-0 text-green-500" />
                                )}
                              </div>
                              <p className="text-muted-foreground text-xs">
                                {article.date}
                              </p>
                            </div>
                          ))}
                        </div>
                      </Card>
                    </div>
                  </div>
                </SheetContent>
              </Sheet>
            </div>

            <div className="grid h-[800px] grid-cols-12 gap-6">
              {/* Desktop Article List */}
              <div className="col-span-4 hidden flex-col gap-4 lg:flex">
                <div className="flex flex-wrap gap-2">
                  <Badge
                    variant={filter === "all" ? "default" : "outline"}
                    className="cursor-pointer px-4 py-2 text-sm transition-colors"
                    onClick={() => setFilter("all")}>
                    All
                  </Badge>
                  <Badge
                    variant={filter === "to_review" ? "default" : "outline"}
                    className="cursor-pointer px-4 py-2 text-sm transition-colors"
                    onClick={() => setFilter("to_review")}>
                    To Review
                  </Badge>
                  <Badge
                    variant={filter === "reviewed" ? "default" : "outline"}
                    className="cursor-pointer px-4 py-2 text-sm transition-colors"
                    onClick={() => setFilter("reviewed")}>
                    Reviewed
                  </Badge>
                </div>

                <Card className="flex flex-1 flex-col overflow-hidden">
                  <CardHeader className="border-b px-4 py-3">
                    <CardTitle className="flex items-center justify-between text-sm font-medium">
                      <span>Articles List</span>
                      <Badge variant="secondary">
                        {filteredArticles?.length} / {labeledArticles.length}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <div className="flex-1 space-y-2 overflow-y-auto p-2">
                    {filteredArticles?.map((article) => (
                      <div
                        key={article.originalIndex}
                        onClick={() =>
                          handleArticleSelect(article.originalIndex)
                        }
                        className={`hover:bg-accent cursor-pointer rounded-lg border p-3 transition-colors ${
                          currentArticle === article.originalIndex
                            ? "bg-accent border-primary"
                            : "bg-card"
                        }`}>
                        <div className="mb-1 flex items-start justify-between">
                          <h4 className="line-clamp-1 text-sm font-medium">
                            {article.title}
                          </h4>
                          {reviewedArticles.has(article.originalIndex) && (
                            <Check className="h-4 w-4 flex-shrink-0 text-green-500" />
                          )}
                        </div>
                        <p className="text-muted-foreground text-xs">
                          {article.date}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              <div className="col-span-12 flex h-full flex-col gap-4 overflow-hidden lg:col-span-8">
                <div className="flex shrink-0 items-center justify-end gap-2">
                  <Button
                    disabled={currentArticle === 0 || isUploading}
                    variant="outline"
                    onClick={handlePrevArticle}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Previous
                  </Button>

                  {currentArticle < labeledArticles.length - 1 ? (
                    <Button
                      disabled={isUploading}
                      variant="outline"
                      onClick={handleNextArticle}>
                      Confirm and go next
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  ) : (
                    <Button
                      disabled={
                        isUploading ||
                        reviewedArticles.size < labeledArticles.length - 1
                      }
                      variant="default"
                      onClick={() => {
                        setReviewedArticles((prev) =>
                          new Set(prev).add(currentArticle)
                        );
                        handleUploadToDatabase();
                      }}>
                      Confirm and Upload
                      <Database className="ml-2 h-4 w-4" />
                    </Button>
                  )}
                </div>

                <Card className="flex h-full flex-col overflow-hidden p-0 lg:flex-row lg:gap-0">
                  <div className="flex w-full flex-col border-b lg:w-[75%] lg:border-r lg:border-b-0">
                    <CardHeader className="shrink-0 border-b py-4">
                      <CardTitle className="text-lg font-bold">
                        {labeledArticles[currentArticle].title}
                      </CardTitle>
                      <CardDescription>
                        {labeledArticles[currentArticle].date}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 overflow-y-auto p-4">
                      <div className="prose dark:prose-invert max-w-none text-sm">
                        {labeledArticles[currentArticle].content
                          .split("\n")
                          .map((str: string, index: number) => (
                            <p key={index} className="mb-2">
                              {str}
                            </p>
                          ))}
                      </div>
                    </CardContent>
                  </div>

                  <div className="flex w-full flex-col lg:w-[25%]">
                    <CardContent className="flex-1 overflow-y-auto p-3">
                      <div className="flex flex-col gap-2">
                        {categories.map((cat) => (
                          <div
                            key={cat}
                            className="flex flex-col gap-1 border-b pb-1 last:border-0">
                            <div className="flex items-center justify-between">
                              <Label className="max-w-[75%] cursor-pointer text-sm leading-tight font-medium break-words">
                                {getCrimeName(cat)}
                              </Label>
                              <Switch
                                checked={
                                  labeledArticles[currentArticle][cat]
                                    ?.value === 1
                                }
                                onCheckedChange={(checked) =>
                                  handleChangeLabel(checked, cat)
                                }
                                className="origin-right scale-75"
                              />
                            </div>
                            <p className="text-muted-foreground text-[10px]">
                              {(
                                (labeledArticles[currentArticle][cat]?.prob ||
                                  0) * 100
                              ).toFixed(1)}
                              %
                            </p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </div>
                </Card>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
export default LabelArticles;
