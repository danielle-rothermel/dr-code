import type { ReactNode } from "react";

export interface GalleryCellProps {
  title: string;
  description: string;
  children: ReactNode;
}

/** Labeled grid cell: component name + variant description + the rendered primitive. */
export function GalleryCell({ title, description, children }: GalleryCellProps) {
  return (
    <figure className="gallery-cell">
      <figcaption className="gallery-cell-caption">
        <span className="gallery-cell-title">{title}</span>
        <span className="gallery-cell-description">{description}</span>
      </figcaption>
      <div className="gallery-cell-body">{children}</div>
    </figure>
  );
}
