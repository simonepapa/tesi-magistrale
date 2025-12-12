const sqlite3 = require("sqlite3");

const dbPath = "../database.db";

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error("Could not connect to database", err);
  } else {
    console.log("Connected to database");
    db.all("SELECT * FROM articles LIMIT 1", (err, rows) => {
      if (err) {
        console.error("Error querying database", err);
      } else {
        console.log("Query result:", rows);
      }
      db.close();
    });
  }
});
