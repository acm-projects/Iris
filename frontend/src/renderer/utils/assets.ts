// Paths to files in frontend/assets (served as Vite's public folder).
//
// Always build image paths with `asset()` instead of writing "/file.png":
// the packaged app loads from file://, where a leading "/" means the root of
// the disk. Vite's base is "./", so these paths stay relative to index.html
// and work both in development and in the packaged app.
export const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;
