// The Iris logo: the glowing flower (same art as the Dock icon).
import { asset } from "../utils/assets";

/**
 * Shown in the sidebar, on the sign-in panel and next to the "Iris" name.
 * Those backgrounds are dark blue in both themes, so one transparent image
 * works everywhere; in light mode it's brightened a little to stand out on
 * the lighter blue (styles.css: .iris-logo).
 */
export default function IrisLogo({ className = "" }: { className?: string }) {
  return (
    <img
      alt=""
      aria-hidden="true"
      className={`iris-logo ${className}`}
      src={asset("brand/iris-logo.png")}
    />
  );
}
