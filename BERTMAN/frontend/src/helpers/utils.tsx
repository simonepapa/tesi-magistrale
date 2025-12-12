export const getCrimeName = (crime: string) => {
  let finalName = "";

  switch (crime) {
    case "omicidio":
      finalName = "Murder";
      break;
    case "omicidio_colposo":
      finalName = "Manslaughter";
      break;
    case "omicidio_stradale":
      finalName = "Road homicide";
      break;
    case "tentato_omicidio":
      finalName = "Attempted murder";
      break;
    case "furto":
      finalName = "Theft";
      break;
    case "rapina":
      finalName = "Robbery";
      break;
    case "violenza_sessuale":
      finalName = "Sexual violence";
      break;
    case "aggressione":
      finalName = "Assault";
      break;
    case "spaccio":
      finalName = "Drug trafficking";
      break;
    case "truffa":
      finalName = "Fraud";
      break;
    case "estorsione":
      finalName = "Extortion";
      break;
    case "contrabbando":
      finalName = "Smuggling";
      break;
    case "associazione_di_tipo_mafioso":
      finalName = "Mafia-type association";
      break;
    default:
      finalName = "Unknown crime";
      break;
  }

  return finalName;
};

export const colorizeSquare = (index: number, palette: string) => {
  let colors: string[] = [];

  if (palette === "red") {
    colors = ["#4a0000", "#b71c1c", "#e57373", "#ffcdd2"];
  } else if (palette === "blue") {
    colors = ["#0d47a1", "#1976d2", "#64b5f6", "#bbdefb"];
  } else if (palette === "green") {
    colors = ["#1b5e20", "#388e3c", "#81c784", "#c8e6c9"];
  }

  return colors[index];
};

export const getQuartiereName = (label: string) => {
  const names: { [key: string]: string } = {
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

  return names[label];
};

export const getQuartiereIndex = (label: string) => {
  const names: { [key: string]: string } = {
    "Bari Vecchia - San Nicola": "bari-vecchia_san-nicola",
    Carbonara: "carbonara",
    Carrassi: "carrassi",
    "Ceglie del Campo": "ceglie-del-campo",
    Japigia: "japigia",
    Libertà: "liberta",
    Loseto: "loseto",
    Madonnella: "madonnella",
    Murat: "murat",
    "Palese - Macchie": "palese-macchie",
    Picone: "picone",
    "San Girolamo - Fesca": "san-girolamo_fesca",
    "San Paolo": "san-paolo",
    "San Pasquale": "san-pasquale",
    "Santo Spirito": "santo-spirito",
    Stanic: "stanic",
    "Torre a mare": "torre-a-mare"
  };

  return names[label];
};
