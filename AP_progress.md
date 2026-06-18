# AP-Fortschritt

Bewertung des Umsetzungsstands
**Status-Legende:**

- ✅ **Erledigt** — im Code/Setup nachweisbar umgesetzt
- 🟡 **Teilweise** — im Ansatz vorhanden, aber unvollständig oder nicht konventionskonform
- ❌ **Offen** — nicht umgesetzt / kein Artefakt vorhanden
- ⚪ **Nicht bewertbar (n. b.)** — organisatorisches oder schriftliches AP, aus `docker/`+`otel/` nicht prüfbar

Stand: 2026-06-18.

---

# 3 Detaillierte Arbeitspakete

## 3.1 AP 1 — Projektmanagement und Organisation

Dieses Arbeitspaket umfasst alle übergeordneten, projektsteuernden Tätigkeiten. Es läuft als Dauer-AP über die gesamte Projektlaufzeit.

| **AP-Code** | **Bezeichnung**        | **Inhalt / Ergebnis**                                                                         | **Status** | **Kommentar** |
| ----------- | ---------------------- | --------------------------------------------------------------------------------------------- | ---------- | ------------- |
| 1.1         | Projektinitialisierung | Themenklärung, Abstimmung mit Betreuer, Formulierung der Forschungsfrage und der Projektziele | ⚪         |               |
| 1.2         | PSP und Zeitplanung    | Erstellung dieses PSP, Ableitung Zeit- und Meilensteinplan, Risikoliste                       | ✅         |               |
| 1.3         | Projekt-Repository     | Einrichtung Git-Repository                                                                    | ✅         |               |
| 1.4         | Laufende Steuerung     | Wöchentliche Selbstreviews, Fortschrittsprotokoll, Nachsteuerung bei Abweichungen             | ⚪         |               |
| 1.5         | Betreuungs-Termine     | Vorbereitung und Nachbereitung von Zwischenbesprechungen mit Betreuer                         | ⚪         |               |
|             |                        |                                                                                               |            |               |

## 3.2 AP 2 — Grundlagen und Recherche

Erarbeitung der konzeptionellen Grundlagen. Ergebnis ist ein Recherchekapitel, das später in die schriftliche Arbeit übernommen wird.

### AP 2.1 Literatur- und Standardrecherche

| **AP-Code** | **Bezeichnung**            | **Inhalt / Ergebnis**                                                                                                              | **Status** | **Kommentar** |
| ----------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------- |
| 2.1.1       | OpenTelemetry-Grundlagen   | Traces, Spans, Events, Metrics, OTLP, Collector, SDK-Architektur                                                                   | 🟡         |               |
| 2.1.2       | GenAI-Semantic-Conventions | Sichtung der aktuellen Spezifikation (Client-, Agent-, Retrieval-Spans, Events, Metrics); Fokus auf Stabilitätsstatus experimental | 🟡         |               |
| 2.1.3       | MLflow 3.x Tracing-Modell  | MLflow-Tracing-Konzepte, OTLP-Export, Dual-Export, OTLP-Ingress                                                                    | 🟡         |               |
| 2.1.4       | RAG-Architektur            | Komponenten eines produktiven RAG-Systems (Chunking, Embeddings, Vektorstore, Retrieval, Prompting, Generation)                    | ✅         |               |

### AP 2.2 Werkzeug-Evaluation

| **AP-Code** | **Bezeichnung**  | **Inhalt / Ergebnis**                                                                                          | **Status** | **Kommentar**           |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------- |
| 2.2.1       | Vektor-Datenbank | Kurze Bewertung Chroma vs. Alternativen; Festlegung auf Chroma                                                 | ✅         | chromadb 1.5.8          |
| 2.2.2       | Embedding-Modell | Auswahl eines eingebetteten Embedding-Modells (lokal, reproduzierbar), Kriterien: Sprache, Lizenz, Modellgröße | ✅         | Chroma-Default-Embedder |
| 2.2.3       | Python-Pakete    | Versionsauswahl opentelemetry-api, -sdk, -semantic-conventions, mlflow, chromadb, LLM-Client                   | ✅         | siehe requirements.txt  |

## 3.3 AP 3 — Aufbau der RAG-Referenzapplikation

Ausbau des bestehenden Mockups zu einer realistischen RAG-Applikation, die als gemeinsame Grundlage für beide Instrumentierungsvarianten dient.

### AP 3.1 Projekt-Setup

| **AP-Code** | **Bezeichnung** | **Inhalt / Ergebnis**                                                        | **Status** | **Kommentar**                                                                                                     |
| ----------- | --------------- | ---------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| 3.1.1       | Projektstruktur | Python-Projekt                                                               | ✅         | Python-Projekt unter `otel/` (App, Seed, Scorer, Runner-Skript).                                                  |
| 3.1.2       | Abhängigkeiten  | Pinning der Versionen, lokales Environment                                   | ✅         | venv-Anleitung in `otel/README.md`; Versionen nur als Prosa, kein Lockfile (siehe 2.2.3).                         |
| 3.1.3       | Container-Setup | Dockerfile und docker-compose für App, Chroma, OTel-Collector, MLflow-Server | 🟡         | `docker/compose.yaml` enthält MLflow, Chroma und einen Seed-Job. **Es fehlen: App-Container und ein Dockerfile.** |

### AP 3.2 RAG-Pipeline

| **AP-Code** | **Bezeichnung**        | **Inhalt / Ergebnis**                                      | **Status** | **Kommentar**                                                                                                    |
| ----------- | ---------------------- | ---------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| 3.2.1       | Dokument-Ingestion     | Laden eines Testkorpus, Chunking-Strategie, Normalisierung | ✅         | `seed_weather.py` lädt Open-Meteo-Daten, bildet Wochen-Summaries als Chunks und normalisiert Felder/Wettercodes. |
| 3.2.2       | Embedding-Komponente   | Einbindung des eingebetteten Embedding-Modell              | ✅         | Kein explizites Embedding-Modell/Batch-Encoding; Chroma-Default wird genutzt (siehe 2.2.2).                      |
| 3.2.3       | Chroma-Anbindung       | Persistenter Chroma-Client, Collection-Aufbau              | ✅         | `HttpClient`, `get_or_create_collection`, `add()` in `seed_weather.py`; persistenter Store in `docker/`.         |
| 3.2.4       | Retrieval              | Ähnlichkeitssuche, Top-k                                   | ✅         | Ähnlichkeitssuche mit `TOP_K` umgesetzt                                                                          |
| 3.2.5       | Prompt- und LLM-Aufruf | Prompt-Template, Kontext-Injektion, LLM-Client-Aufruf      | ✅         | `build_prompt` mit Template + Kontext-Injektion; LLM-Aufruf über OpenAI-kompatiblen Endpoint (LM Studio).        |
| 3.2.6       | End-to-End-Query-Flow  | Query-Funktion: Frage → Retrieval → Prompting → Antwort    | ✅         | `rag_mock` deckt den kompletten Flow ab.                                                                         |

## 3.4 AP 4 — Instrumentierung mit MLflow-nativem Tracing

Erste von zwei Instrumentierungsvarianten: Die RAG-Applikation wird mit den MLflow-eigenen Tracing-Mechanismen versehen.

| **AP-Code** | **Bezeichnung**              | **Inhalt / Ergebnis**                                                                         | **Status** | **Kommentar**                                                                                          |
| ----------- | ---------------------------- | --------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| 4.1         | MLflow-Setup                 | MLflow-Tracking-Server lokal aufsetzen, Experiment anlegen, Autologging prüfen                | ✅         | Tracking-Server in `docker/compose.yaml` (Port 5000, SQLite-Backend); Experiment via `set_experiment`. |
| 4.2         | Manuelle Spans               | Instrumentierung der RAG-Schritte mittels mlflow.trace / @mlflow.trace-Decorator              | ✅         | `@mlflow.trace` + `mlflow.start_span` je Schritt (`otel/rag.py` als reine Variante A).                 |
| 4.3         | Attribute und Inputs/Outputs | Erfassung von Query, Retrieved Docs, Prompt, Response, Modellparametern                       | ✅         | `set_inputs`/`set_outputs`/`set_attribute` für Query, Docs, Prompt, Antwort, Tokens, Modell.           |
| 4.4         | OTLP-Export aus MLflow       | Konfiguration des OTLP-Exports an den OTel-Collector, Prüfung Dual-Export **(TO BE DECIDED)** | ✅         | vi MLFlow OTLP-ingress                             |
| 4.5         | Qualitäts-Checks             | Nutzung der MLflow-Qualitätsprüfungen/Evaluation für Traces (Mlflow.evaluate für GenAI)       | ✅         | `scorer.py`: `trace_schema_health` + `mlflow.genai.evaluate`                                           |
| 4.6         | Dokumentation Variante A     | Kurzbeschreibung Aufbau, Screenshot des MLflow-UI, Ablageort der Trace-Exports                | ❌         | Keine Kurzbeschreibung, kein Screenshot, kein dokumentierter Ablageort der Trace-Exports.              |

## 3.5 AP 5 — Instrumentierung mit OpenTelemetry-SDK

Zweite Instrumentierungsvariante: Dieselbe RAG-Applikation wird mit reinem OpenTelemetry-SDK versehen. Die Instrumentierung folgt strikt den GenAI-Semantic-Conventions.

### AP 5.1 OTel-Grundgerüst

| **AP-Code** | **Bezeichnung**         | **Inhalt / Ergebnis**                                                               | **Status** | **Kommentar**                                                                                                                                                                          |
| ----------- | ----------------------- | ----------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1.1       | SDK-Konfiguration       | TracerProvider, Resource-Attribute (service.name, service.version), Sampler         | 🟡         | `TracerProvider` gesetzt. Aber: `service.name` nur als Tracer-/Scope-Name (`get_tracer(...)`), **nicht als `Resource`-Attribut**; `service.version` und ein expliziter Sampler fehlen. |
| 5.1.2       | OTLP-Exporter           | OTLP-gRPC/HTTP-Exporter, BatchSpanProcessor, Verbindung zum Collector               | ❌         | .                      |
| 5.1.3       | Collector-Konfiguration | OTel-Collector-Pipeline: Receiver (OTLP) → Processor → Exporter **(TO BE DECIDED)** | ❌         | Kein OTel-Collector, wir exportieren direkt an MLflow.                                                                                                                                 |

### AP 5.2 Konventionskonforme Spans

| **AP-Code** | **Bezeichnung**             | **Inhalt / Ergebnis**                                                                  | **Status** | **Kommentar**                                                                                                                                                                   |
| ----------- | --------------------------- | -------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.2.1       | Client-Spans für LLM-Aufruf | gen_ai._ Attribute: system, request.model, request.parameters, response.model, usage._ | ✅         | `lmstudio_inference.otel` mit `gen_ai.operation.name`, `provider.name`, `request.model`, `request.temperature`, `usage.input/output_tokens`, `response.model/id/finish_reason`. |
| 5.2.2       | Retrieval-Spans             | Umsetzung der seit Feb 2026 neuen Retrieval-Span-Konvention für die Chroma-Anfrage     | ✅         | `chroma_retrieval.otel` mit `gen_ai.operation.name=retrieval`, `data_source.id`, `server.*` und Hit-Events. An neuer Konvention orientiert.                                     |
| 5.2.3       | Agent-/Workflow-Span        | Umschließender Span für den End-to-End-Query-Flow gemäß Agent-Span-Konvention          | 🟡         | Umschließender `rag_request.otel`-Span vorhanden, aber `kind=INTERNAL` statt der Agent-Span-Konvention (z. B. `gen_ai.operation.name`/`agent`-Attribute fehlen).                |
| 5.2.4       | Events                      | GenAI-Events (z. B. user/assistant/tool messages) gemäß Event-Konvention               | ✅         | `add_event` für `retrieval.hit`, `gen_ai.request.summary`, `gen_ai.response.summary`. Strenge Event-Namenskonvention (user/assistant) noch nicht voll abgebildet.               |
| 5.2.5       | Metrics                     | Token-Usage-, Latenz- und Fehler-Metriken gemäß GenAI-Metric-Konvention                | 🟡         |  console als json, nicht mlflow kompatibel                                    |

## 3.6 AP 6 — Evaluation, Vergleich und Erweiterbarkeit (optional)

Zusammenführung der beiden Varianten und Beantwortung der Forschungsfragen.

### AP 6.1 Vergleich der Instrumentierungen

| **AP-Code** | **Bezeichnung**         | **Inhalt / Ergebnis**                                                                                     | **Status** | **Kommentar**                                                                                         |
| ----------- | ----------------------- | --------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| 6.1.1       | Vergleichskriterien     | Ableitung eines Kriterienkatalogs: Konventionstreue, Aufwand, Lesbarkeit, Tool-Abhängigkeit, Portabilität | ❌         | Kein Kriterienkatalog                                                                                 |
| 6.1.2       | Trace-Strukturvergleich | Gegenüberstellung der Spans/Attribute/Events aus Variante A und B an identischen Queries                  | ❌         | Grundlage vorhanden (`rag.py` vs. `rag-otel.py`), aber kein Vergleichsartefakt/Gegenüberstellung.     |
| 6.1.3       | Konventionskonformität  | Pro Variante: Coverage der GenAI-Semconv (Attribute, Events, Metrics)                                     | ❌         | Keine Coverage-Auswertung pro Variante.                                                               |
| 6.1.4       | MLflow-Qualitätsprüfung | Ausführung der MLflow-Qualitäts-Checks gegen beide Varianten (Mlflow.evaluate für GenAI)                  | 🟡         | Scorer (`scorer.py`) existiert, aber nicht dokumentiert gegen beide Varianten ausgeführt/ausgewertet. |

### AP 6.2 Erweiterbarkeit

| **AP-Code** | **Bezeichnung** | **Inhalt / Ergebnis**                                                                  | **Status** | **Kommentar**                                                                                                           |
| ----------- | --------------- | -------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| 6.2.1       | Domain-Events   | Modellierung mindestens eines applikationsspezifischen Events und Bindung an den Trace | 🟡         | `retrieval.hit`-Events vorhanden, aber nicht bewusst als applikationsspezifisches Domain-Event modelliert/dokumentiert. |
| 6.2.2       | Anwendungslogs  | Logs-to-Traces-Korrelation (trace_id/span_id-Injection), OTLP-Log-Export               | ❌         | Keine Logs-to-Traces-Korrelation, kein OTLP-Log-Export.                                                                 |
| 6.2.3       | Bewertung       | Bewertung: Welche Option bricht die Konvention, welche ergänzt sie sauber?             | ❌         | Keine Bewertung vorhanden.                                                                                              |

## 3.7 AP 7 — Dokumentation und Projektabschluss

| **AP-Code** | **Bezeichnung**                     | **Inhalt / Ergebnis**                                                   | **Status** | **Kommentar**                                                                                                                |
| ----------- | ----------------------------------- | ----------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 7.1         | Schriftliche Arbeit — Rohfassung    | Struktur, Einleitung, Grundlagen, Implementierung, Ergebnisse           | ⚪         | TBD                                                                                                                          |
| 7.2         | Schriftliche Arbeit — Überarbeitung | Sprache, Abbildungen, Tabellen, Literaturverzeichnis                    | ⚪         | TBD                                                                                                                          |
| 7.3         | Code-Dokumentation                  | README mit Setup-Anleitung, Architektur-Diagramm, Reproduktionsschritte | 🟡         | `otel/README.md` deckt nur Setup ab (mit Tippfehlern); kein Architektur-Diagramm, keine vollständigen Reproduktionsschritte. |
| 7.4         | Abschlusspräsentation               | Foliensatz zu Ergebnissen und Demo, zum Vorstellen am Projekttag        | ⚪         |                                                                                                                              |
| 7.5         | Abgabe                              | Abgabe der Projektarbeit und ggf. des GitHub-Repos                      | ⚪         |                                                                                                                              |

# 4 Meilensteinplan

Der Zeitplan ist auf ca. 8 Wochen Projektlaufzeit ausgelegt. Die Wochenangaben verstehen sich relativ zum Projektstart.

| **MS** | **Meilenstein**                   | **Zeitpunkt (rel.)** | **Ergebnis / Kriterium**                                      | **Status** | **Kommentar**                                                                                |
| ------ | --------------------------------- | -------------------- | ------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------- |
| M1     | Projektdefinition abgeschlossen   | Ende Woche 1         | PSP, Zeitplan, Repository stehen (APs 1.1 – 1.3)              | ✅         | PSP und Git-Repository liegen vor.                                                           |
| M2     | Grundlagen abgeschlossen          | Ende Woche 2         | Recherche- und Werkzeug-Auswahl liegen vor (AP 2)             | 🟡         | Werkzeugauswahl (Chroma) getroffen; Embedding-Auswahl und Recherche-Niederschrift offen.     |
| M3     | RAG-Referenzapplikation lauffähig | Ende Woche 4         | Rag-pipeline funktioniert, Chroma liefert Antworten (AP 3)    | ✅         | End-to-End-Flow lauffähig; Container-Set unvollständig (3.1.3).                              |
| M4     | Variante A (MLflow) fertig        | Mitte Woche 5        | Traces im MLflow-UI sichtbar, OTLP-Export funktioniert (AP 4) | 🟡         | Traces erzeugt; OTLP-Export (4.4) offen, Doku Variante A (4.6) fehlt.                        |
| M5     | Variante B (OTel) fertig          | Ende Woche 6         | Spans, Events und Metrics gemäß GenAI-Semconv erzeugt (AP 5)  | 🟡         | Spans/Events vorhanden, aber **Metrics + OTLP/Collector fehlen** → Variante B unvollständig. |
| M6     | Evaluation abgeschlossen          | Mitte Woche 7        | Vergleich und Erweiterbarkeits-Analyse liegen vor (AP 6)      | ❌         | Noch nicht begonnen (kein Vergleichsartefakt).                                               |
| M7     | Projektabgabe                     | Ende Woche 8         | Code und schriftliche Arbeit abgegeben (AP 7)                 | ❌         | Steht aus.                                                                                   |

# 5 Aufwandsübersicht nach Hauptarbeitspaket

| **AP** | **Hauptarbeitspaket**                       | **Status** | **Kommentar**                                                           |
| ------ | ------------------------------------------- | ---------- | ----------------------------------------------------------------------- |
| AP 1   | Projektmanagement und Organisation          | 🟡         | Repo (1.3) ✅; übrige PM-Artefakte nicht aus Code prüfbar.              |
| AP 2   | Grundlagen und Recherche                    | 🟡         | Umsetzung belegt Recherche; Embedding-Auswahl + Niederschrift offen.    |
| AP 3   | Aufbau der RAG-Referenzapplikation          | 🟡         | Pipeline ✅; Container-Set + explizite Embedding-Komponente offen.      |
| AP 4   | Instrumentierung mit MLflow-nativem Tracing | 🟡         | Kern (4.1–4.3, 4.5) ✅; OTLP-Export (4.4) + Doku (4.6) offen.           |
| AP 5   | Instrumentierung mit OpenTelemetry-SDK      | 🟡         | Spans/Events ✅; Resource-Attrs, Exporter/Collector, **Metrics** offen. |
| AP 6   | Evaluation, Vergleich und Erweiterbarkeit   | ❌         | Kaum begonnen; Scorer als Baustein für 6.1.4 vorhanden.                 |
| AP 7   | Dokumentation und Projektabschluss          | 🟡         | Setup-README vorhanden; Diagramm/Arbeit/Abgabe offen.                   |

---

## Zusammenfassung der wichtigsten Lücken (priorisiert)

1. **OTel-Metrics (5.2.5) + OTLP-Exporter/Collector (5.1.2/5.1.3, 4.4)** — kritischste Lücke. Variante B ist derzeit „nur Spans, über MLflow geroutet" und damit kein reines OpenTelemetry-SDK-Setup. Liegt auf dem kritischen Pfad der Forschungsfrage.
2. **Resource-Attribute (5.1.1)** — `service.name`/`service.version` nicht als `Resource` gesetzt; relevant für die bewertete Konventionstreue (6.1.3).
3. **Vergleichsartefakt (AP 6 / M6)** — Kern der Arbeit; Grundlage (A vs. B) existiert, aber kein Kriterienkatalog/Strukturvergleich/Coverage-Tabelle.
4. **Container-Set (3.1.3)** — kein Dockerfile, kein Collector-Service, App läuft per `run.sh` auf dem Host.
5. **Dokumentation/Reproduzierbarkeit (7.3, 3.1.2)** — kein `requirements.txt`, kein Architektur-Diagramm, knappe README.

## Stärken

- `scorer.py` (`trace_schema_health` + `mlflow.genai.evaluate`) ist über den Plan hinaus ausgearbeitet und bedient 4.5 sowie die Grundlage für 6.1.4.
- Breite, konventionsbewusste `gen_ai.*`-Span-/Attribut-Abdeckung (5.2.1/5.2.2/5.2.4).
- Saubere Trennung der Varianten (`rag.py` MLflow-only vs. `rag-otel.py`) als valide A/B-Basis für identische Queries.
