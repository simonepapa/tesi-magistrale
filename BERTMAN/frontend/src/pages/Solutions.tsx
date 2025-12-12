import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import {
  Building2,
  Camera,
  Ear,
  ChevronsUpDown,
  Gavel,
  Globe,
  Lightbulb,
  MessageSquare,
  Radio,
  ShieldAlert,
  Users,
  Video
} from "lucide-react";
import { useState } from "react";

type CrimeType = "murder" | "theft" | "assault" | "drugs" | "organized-crime";

interface Solution {
  id: string;
  title: string;
  description: string;
  detailedDescription?: string;
  benefits?: string[];
  implementation?: string;
  considerations?: string;
  icon: React.ElementType;
  crimeTypes: CrimeType[];
}

const solutions: Solution[] = [
  {
    id: "smart-surveillance",
    title: "Smart Surveillance Systems",
    description:
      "Use CCTV cameras equipped with facial recognition and behavioral analysis. These systems are useful to identify unusual movements and send real-time alerts to law enforcement. Disclaimer: remember to respect privacy laws.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Video,
    crimeTypes: ["murder"]
  },
  {
    id: "acoustic-detection",
    title: "Acoustic Detection",
    description:
      "Use IoT-enabled acoustic sensors to detect gunshots, screams or sounds of distress. These sensors, using machine learning, are able to identify an unusual sound from a typical sound and notify authorities.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Ear,
    crimeTypes: ["murder", "assault"]
  },
  {
    id: "traffic-monitoring",
    title: "Traffic Monitoring Systems",
    description:
      "These systems are particularly useful to take a photo or recognize a license plate to identify murderers who flee the scene of a road homicide.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Camera,
    crimeTypes: ["murder"]
  },
  {
    id: "anti-theft",
    title: "Smart Anti-Theft Systems",
    description:
      "Promote the installation of IoT-enabled alarm systems in residential and commercial spaces to automatically alert authorities in case of unauthorized entry.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: ShieldAlert,
    crimeTypes: ["theft"]
  },
  {
    id: "street-lighting",
    title: "Intelligent Street Lighting",
    description:
      "Install motion-sensitive LED lighting systems which illuminate when movement is detected, deterring criminal activities and making public spaces safer.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Lightbulb,
    crimeTypes: ["theft", "assault"]
  },
  {
    id: "panic-buttons",
    title: "Emergency Panic Buttons",
    description:
      "Install panic buttons in strategic locations like bus stops, parks and public restrooms which, when activated, these buttons alert nearby patrol units and the city's emergency center, providing precise GPS coordinates.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Radio,
    crimeTypes: ["assault"]
  },
  {
    id: "crowd-monitoring",
    title: "Crowd Flow Monitoring",
    description:
      "Utilize anonymized mobile device data to monitor crowd movements and identify irregular patterns that could indicate potential assaults. Heat maps can assist in preemptively addressing risky zones.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Users,
    crimeTypes: ["assault"]
  },
  {
    id: "digital-monitoring",
    title: "Digital Communication Monitoring",
    description:
      "Deploy AI-driven tools to analyze social media, messaging platforms and the dark web to identify patterns associated with drug trafficking or fraudulent schemes.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: MessageSquare,
    crimeTypes: ["drugs"]
  },
  {
    id: "blockchain",
    title: "Blockchain Technology",
    description:
      "Encourage the use of blockchain for secure digital transactions to reduce financial fraud. Smart contracts can provide transparency and accountability in financial dealings.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Gavel,
    crimeTypes: ["drugs", "organized-crime"]
  },
  {
    id: "transaction-anomaly",
    title: "Transaction Anomaly Detection",
    description:
      "Use financial analytics software to monitor unusual patterns in transactions, such as large cash withdrawals, frequent small deposits or suspicious money transfers that may indicate extortion or laundering.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Building2,
    crimeTypes: ["organized-crime"]
  },
  {
    id: "collaborative-platforms",
    title: "Collaborative Platforms",
    description:
      "Implement platforms where law enforcement, local governments and private companies can share intelligence securely in order to facilitate coordinated efforts to dismantle organized crime networks.",
    detailedDescription:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    benefits: [
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
      "Sed do eiusmod tempor incididunt ut labore et dolore",
      "Quis nostrud exercitation ullamco laboris nisi ut aliquip"
    ],
    implementation:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    considerations:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    icon: Globe,
    crimeTypes: ["organized-crime", "drugs"]
  }
];

const crimeFilters = [
  { id: "all" as const, label: "All" },
  { id: "murder" as CrimeType, label: "Murder & Violent Crimes" },
  { id: "theft" as CrimeType, label: "Theft & Robbery" },
  { id: "assault" as CrimeType, label: "Sexual Violence & Assault" },
  { id: "drugs" as CrimeType, label: "Drug Trafficking & Fraud" },
  { id: "organized-crime" as CrimeType, label: "Organized Crime" }
];

function Solutions() {
  const [activeFilters, setActiveFilters] = useState<CrimeType[]>([]);
  const [selectedSolution, setSelectedSolution] = useState<Solution | null>(
    null
  );
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const toggleFilter = (filter: CrimeType | "all") => {
    if (filter === "all") {
      setActiveFilters([]);
      return;
    }

    setActiveFilters((prev) =>
      prev.includes(filter)
        ? prev.filter((f) => f !== filter)
        : [...prev, filter]
    );
  };

  const isFilterActive = (filter: CrimeType | "all") => {
    if (filter === "all") {
      return activeFilters.length === 0;
    }
    return activeFilters.includes(filter);
  };

  const handleCardClick = (solution: Solution) => {
    setSelectedSolution(solution);
    setIsDialogOpen(true);
  };

  const handleDialogClose = () => {
    setIsDialogOpen(false);
    setTimeout(() => setSelectedSolution(null), 200); // Wait for dialog animation
  };

  const filteredSolutions =
    activeFilters.length === 0
      ? solutions
      : solutions.filter((solution) =>
          solution.crimeTypes.some((type) => activeFilters.includes(type))
        );

  return (
    <div className="mt-8 mb-12 flex max-w-[1200px] flex-col gap-6 px-4 lg:mx-auto xl:px-0">
      <h1 className="text-4xl font-bold">
        Smart solutions and strategies to improve Urban Security
      </h1>

      <div className="flex flex-wrap gap-2">
        {crimeFilters.map((filter) => (
          <Badge
            key={filter.id}
            variant={isFilterActive(filter.id) ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm transition-colors"
            onClick={() => toggleFilter(filter.id)}>
            {filter.label}
          </Badge>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
        {filteredSolutions.map((solution) => {
          const Icon = solution.icon;
          return (
            <Card
              key={solution.id}
              className="group bg-card/50 hover:bg-card/80 flex cursor-pointer flex-col transition-all hover:shadow-lg"
              onClick={() => handleCardClick(solution)}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Icon className="text-primary h-6 w-6" />
                  {solution.title}
                  <ChevronsUpDown className="text-muted-foreground group-hover:text-primary ml-auto h-4 w-4 transition-colors" />
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <p className="text-muted-foreground line-clamp-3 text-sm">
                  {solution.description}
                </p>
              </CardContent>
              <CardFooter>
                <div className="flex flex-wrap gap-1">
                  {solution.crimeTypes.map((type) => (
                    <Badge key={type} variant="secondary" className="text-xs">
                      {crimeFilters.find((f) => f.id === type)?.label || type}
                    </Badge>
                  ))}
                </div>
              </CardFooter>
            </Card>
          );
        })}
      </div>

      {filteredSolutions.length === 0 && (
        <div className="text-muted-foreground py-12 text-center">
          No solutions found for the selected filters.
        </div>
      )}

      <Dialog open={isDialogOpen} onOpenChange={handleDialogClose}>
        <DialogContent className="bg-card max-h-[85vh] w-[80%] overflow-y-auto rounded-lg lg:max-w-3xl">
          {selectedSolution && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-3 text-2xl">
                  {(() => {
                    const Icon = selectedSolution.icon;
                    return <Icon className="text-primary h-8 w-8" />;
                  })()}
                  {selectedSolution.title}
                </DialogTitle>
                <DialogDescription className="flex flex-wrap gap-1 pt-2">
                  {selectedSolution.crimeTypes.map((type) => (
                    <Badge key={type} variant="secondary" className="text-xs">
                      {crimeFilters.find((f) => f.id === type)?.label || type}
                    </Badge>
                  ))}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6 pt-4">
                <section>
                  <h3 className="mb-2 text-lg font-semibold">Overview</h3>
                  <p className="text-muted-foreground">
                    {selectedSolution.detailedDescription ||
                      selectedSolution.description}
                  </p>
                </section>

                {selectedSolution.benefits &&
                  selectedSolution.benefits.length > 0 && (
                    <section>
                      <h3 className="mb-2 text-lg font-semibold">
                        Key Benefits
                      </h3>
                      <ul className="list-inside list-disc space-y-2">
                        {selectedSolution.benefits.map((benefit, index) => (
                          <li key={index} className="text-muted-foreground">
                            {benefit}
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}

                {selectedSolution.implementation && (
                  <section>
                    <h3 className="mb-2 text-lg font-semibold">
                      Implementation Guidelines
                    </h3>
                    <p className="text-muted-foreground">
                      {selectedSolution.implementation}
                    </p>
                  </section>
                )}

                {selectedSolution.considerations && (
                  <section>
                    <h3 className="mb-2 text-lg font-semibold">
                      Important Considerations
                    </h3>
                    <p className="text-muted-foreground">
                      {selectedSolution.considerations}
                    </p>
                  </section>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
export default Solutions;
