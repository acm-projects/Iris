/** Shared, intentionally blank Iris page canvas for features still to be built. */
export function PagePlaceholder({ page }: { page: string }) {
  return (
    <section className="blank-page">
      <div>
        <p className="page-kicker">IRIS</p>
        <h1>{page}</h1>
        <p>This workspace is ready for the next Iris feature.</p>
      </div>
    </section>
  );
}
