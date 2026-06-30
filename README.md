# NYAAI Lecture Prep

Separate frontend and backend apps for AI lecture generation from an ingested PDF knowledge base.

## Folders

- `knowledge-base/` - Node.js/Express backend and RAG lecture API.
- `frontend/` - Vite React frontend connected to the backend API.

## Backend

```bash
cd knowledge-base
npm install
cp .env.example .env
npm run dev
```

The backend runs on `http://localhost:3000`.

## Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

The frontend runs on `http://localhost:5173` and proxies `/api` calls to the backend in development.

## Notes

- Do not commit real `.env` files.
- PDFs are ingested through the backend CLI only. The frontend intentionally has no upload flow.
