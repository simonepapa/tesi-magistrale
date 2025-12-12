import { Router, Request, Response } from "express";
import sqlite3 from "sqlite3";
import fs from "fs";
import { analyze_quartieri, calculate_statistics } from "../utils";

const router = Router();
const dbPath = "../database.db";
const quartieriJsonPath = "../classifier/data/quartieri.json";

// Cache quartieri.json in memory as it is static and not expected to change
let quartieriGeometryCache: any = null;

function loadQuartieriGeometry() {
  if (!quartieriGeometryCache) {
    try {
      const data = fs.readFileSync(quartieriJsonPath, "utf8");
      quartieriGeometryCache = JSON.parse(data);
      console.log("✅ Loaded quartieri.json into cache");
    } catch (err) {
      console.error("❌ Failed to load quartieri.json:", err);
      throw err;
    }
  }
  return quartieriGeometryCache;
}

// Load quartieri data at server startup
loadQuartieriGeometry();

// Helper to get DB connection
const getDb = () => {
  return new sqlite3.Database(dbPath, (err) => {
    if (err) {
      console.error("Could not connect to database", err);
    }
  });
};

router.get("/get-data", (req: Request, res: Response) => {
  let startDate = (req.query.startDate as string) || "";
  let endDate = (req.query.endDate as string) || "";
  let crimes = (req.query.crimes as string) || "";
  let quartieri = (req.query.quartieri as string) || "";
  const weightsForArticles = req.query.weightsForArticles !== "false";
  const weightsForPeople = req.query.weightsForPeople === "true";
  const minmaxScaler = req.query.minmaxScaler !== "false";

  // Validate and normalize dates
  const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
  if (startDate) {
    // Extract just the date part if format is "YYYY-MM-DD HH:mm:ss"
    const parts = startDate.split(" ");
    startDate = parts[0] || startDate;
    if (!dateRegex.test(startDate)) {
      res
        .status(400)
        .json({ error: "Invalid startDate format. Expected YYYY-MM-DD" });
      return;
    }
  }
  if (endDate) {
    // Extract just the date part if format is "YYYY-MM-DD HH:mm:ss"
    const parts = endDate.split(" ");
    endDate = parts[0] || endDate;
    if (!dateRegex.test(endDate)) {
      res
        .status(400)
        .json({ error: "Invalid endDate format. Expected YYYY-MM-DD" });
      return;
    }
  }

  // Define valid crimes and quartieri
  const validCrimes = [
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
  const validQuartieri = [
    "bari-vecchia_san-nicola",
    "carbonara",
    "carrassi",
    "ceglie-del-campo",
    "japigia",
    "liberta",
    "loseto",
    "madonnella",
    "murat",
    "palese-macchie",
    "picone",
    "san-paolo",
    "san-pasquale",
    "santo-spirito",
    "stanic",
    "torre-a-mare",
    "san-girolamo_fesca"
  ];

  if (
    !crimes ||
    (crimes.split(",").length <= 1 && crimes.split(",")[0] === "")
  ) {
    crimes = validCrimes.join(",");
  } else {
    // Validate crimes
    const crimesList = crimes.split(",");
    const invalidCrimes = crimesList.filter((c) => !validCrimes.includes(c));
    if (invalidCrimes.length > 0) {
      res
        .status(400)
        .json({ error: `Invalid crimes: ${invalidCrimes.join(", ")}` });
      return;
    }
  }

  if (
    !quartieri ||
    (quartieri.split(",").length <= 1 && quartieri.split(",")[0] === "")
  ) {
    quartieri = validQuartieri.join(",");
  } else {
    // Validate quartieri
    const quartieriList = quartieri.split(",");
    const invalidQuartieri = quartieriList.filter(
      (q) => !validQuartieri.includes(q)
    );
    if (invalidQuartieri.length > 0) {
      res
        .status(400)
        .json({ error: `Invalid quartieri: ${invalidQuartieri.join(", ")}` });
      return;
    }
  }

  const crimesList = crimes.split(",");
  const quartieriList = quartieri.split(",");

  const quartieri_data = quartieriList.map((quartiere) => ({
    Quartiere: quartiere,
    "Totale crimini": 0,
    "Indice di rischio": 0,
    "Indice di rischio normalizzato": 0
  }));

  // Read all articles from the database
  const db = getDb();
  let query = "SELECT * FROM articles";
  const params: any[] = [];

  if (startDate && endDate) {
    query += " WHERE date BETWEEN ? AND ?";
    params.push(startDate, endDate);
  }

  db.all(query, params, (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
      db.close();
      return;
    }

    const articles_df = rows;

    try {
      const geometry_json = loadQuartieriGeometry();

      let geojson_data = {
        type: "FeatureCollection",
        features: [] as any[]
      };

      quartieriList.forEach((quartiere) => {
        const matching_quartiere = geometry_json.find(
          (feature: any) => feature.python_id === quartiere
        );

        if (matching_quartiere) {
          geojson_data.features.push({
            type: "Feature",
            properties: {
              name: matching_quartiere.name,
              python_id: matching_quartiere.python_id
            },
            geometry: matching_quartiere.geometry
          });
        }
      });

      // Analysis
      const analyzed_geojson = analyze_quartieri(
        articles_df,
        quartieri_data,
        geojson_data,
        crimesList,
        weightsForArticles,
        weightsForPeople,
        minmaxScaler
      );

      // Statistics
      const final_geojson = calculate_statistics(
        quartieri_data,
        analyzed_geojson
      );

      res.json(final_geojson);
      db.close();
    } catch (err) {
      res.status(500).json({ error: "Could not read quartieri.json" });
      db.close();
      return;
    }
  });
});

router.get("/get-articles", (req: Request, res: Response) => {
  const quartiere = req.query.quartiere as string;
  const startDate = req.query.startDate as string;
  const endDate = req.query.endDate as string;
  const page = req.query.page ? parseInt(req.query.page as string) : undefined;
  const limit = req.query.limit
    ? parseInt(req.query.limit as string)
    : undefined;

  const db = getDb();
  let query = "SELECT * FROM articles";
  let countQuery = "SELECT COUNT(*) as total FROM articles";
  const params: any[] = [];
  const conditions: string[] = [];

  if (quartiere) {
    conditions.push("quartiere = ?");
    params.push(quartiere);
  }

  if (startDate && endDate) {
    conditions.push("date BETWEEN ? AND ?");
    params.push(startDate, endDate);
  }

  if (conditions.length > 0) {
    const conditionStr = " WHERE " + conditions.join(" AND ");
    query += conditionStr;
    countQuery += conditionStr;
  }

  query += " ORDER BY date DESC";

  if (page && limit) {
    const offset = (page - 1) * limit;
    query += ` LIMIT ? OFFSET ?`;
    params.push(limit, offset);

    // For count query, we need the params without limit/offset
    // We can't reuse 'params' directly because we just pushed limit/offset
    // So we need to use a separate params array for the count query or slice the current one
    const countParams = params.slice(0, params.length - 2);

    db.get(countQuery, countParams, (err, row: any) => {
      if (err) {
        res.status(500).json({ error: err.message });
        db.close();
        return;
      }

      const total = row.total;

      db.all(query, params, (err, rows) => {
        if (err) {
          res.status(500).json({ error: err.message });
        } else {
          res.json({
            articles: rows,
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit)
          });
        }
        db.close();
      });
    });
  } else {
    // Legacy behavior for Dashboard (return all articles)
    db.all(query, params, (err, rows) => {
      if (err) {
        res.status(500).json({ error: err.message });
      } else {
        res.json(rows);
      }
      db.close();
    });
  }
});

router.post("/upload-to-database", (req: Request, res: Response) => {
  const { jsonFile } = req.body;

  if (!jsonFile || !Array.isArray(jsonFile)) {
    res
      .status(400)
      .json({ error: "Invalid input: 'jsonFile' must be an array" });
    return;
  }

  const requiredFields = ["link", "title", "date", "content", "python_id"];
  const validCrimes = [
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

  for (let i = 0; i < jsonFile.length; i++) {
    const item = jsonFile[i];
    if (typeof item !== "object" || item === null) {
      res
        .status(400)
        .json({ error: `Invalid input: Item at index ${i} is not an object` });
      return;
    }

    const missingFields = requiredFields.filter((field) => !(field in item));
    if (missingFields.length > 0) {
      res.status(400).json({
        error: `Invalid input: Item at index ${i} is missing fields: ${missingFields.join(", ")}`
      });
      return;
    }

    for (const crime of validCrimes) {
      if (
        !item[crime] ||
        typeof item[crime].value !== "number" ||
        typeof item[crime].prob !== "number"
      ) {
        res.status(400).json({
          error: `Invalid input: Item at index ${i} has invalid or missing crime label '${crime}'`
        });
        return;
      }
    }
  }

  const db = getDb();
  const stmt = db.prepare(`
    INSERT INTO articles (link, quartiere, title, date, content, omicidio, omicidio_prob, omicidio_colposo, omicidio_colposo_prob, omicidio_stradale, omicidio_stradale_prob, tentato_omicidio, tentato_omicidio_prob, furto, furto_prob, rapina, rapina_prob, violenza_sessuale, violenza_sessuale_prob, aggressione, aggressione_prob, spaccio, spaccio_prob, truffa, truffa_prob, estorsione, estorsione_prob, contrabbando, contrabbando_prob, associazione_di_tipo_mafioso, associazione_di_tipo_mafioso_prob)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  db.serialize(() => {
    jsonFile.forEach((item: any) => {
      stmt.run(
        item.link || "",
        item.python_id || "",
        item.title || "",
        item.date || "",
        item.content || "",
        item.omicidio?.value,
        item.omicidio?.prob,
        item.omicidio_colposo?.value,
        item.omicidio_colposo?.prob,
        item.omicidio_stradale?.value,
        item.omicidio_stradale?.prob,
        item.tentato_omicidio?.value,
        item.tentato_omicidio?.prob,
        item.furto?.value,
        item.furto?.prob,
        item.rapina?.value,
        item.rapina?.prob,
        item.violenza_sessuale?.value,
        item.violenza_sessuale?.prob,
        item.aggressione?.value,
        item.aggressione?.prob,
        item.spaccio?.value,
        item.spaccio?.prob,
        item.truffa?.value,
        item.truffa?.prob,
        item.estorsione?.value,
        item.estorsione?.prob,
        item.contrabbando?.value,
        item.contrabbando?.prob,
        item.associazione_di_tipo_mafioso?.value,
        item.associazione_di_tipo_mafioso?.prob
      );
    });
    stmt.finalize();
  });

  db.close((err) => {
    if (err) {
      res.status(500).send("Error uploading to database");
    } else {
      res.send("Uploaded file succesfully");
    }
  });
});

export default router;
