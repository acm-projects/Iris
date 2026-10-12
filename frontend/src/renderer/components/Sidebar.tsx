// Sidebar shown on every signed-in page.
import type { ChangeEvent } from "react";
import { asset } from "../utils/assets";
// Pages in the order they appear. Each name maps to an icon in
// assets/sidebar-icons/<name>.svg (and <name>-active.svg when selected).
const nav = ["Home", "Calendar", "Translation", "Settings"];

/**
 * Left navigation: page links, and the profile (photo, name, email) at the
 * bottom. Signing out and changing
 * the photo live in Settings; clicking the profile opens it.
 */
export default function Sidebar({
  active,
  avatarUrl,
  email,
  fileInput,
  name,
  onAvatarChange,
  onNavigate,
}: {
  active: string;
  avatarUrl: string | null;
  email: string;
  fileInput: React.RefObject<HTMLInputElement | null>;
  name: string;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onNavigate: (page: string) => void;
}) {
  return (
    <aside className="sidebar">
      <a className="wordmark" href="#home">
        <img src={asset("iris-mark.png")} alt="" />
        <span>Iris</span>
      </a>
      {/* Page links; the active page gets the gold icon and highlight */}
      <nav>
        {nav.map((item) => (
          <button
            className={active === item ? "active" : ""}
            key={item}
            onClick={() => onNavigate(item)}
            type="button"
          >
            <Icon active={active === item} item={item} />
            <em>{item}</em>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        {/* Profile: photo, name and email; opens Settings */}
        <button
          className="user-chip"
          onClick={() => onNavigate("Settings")}
          title="Account settings"
          type="button"
        >
          {avatarUrl ? (
            <img src={avatarUrl} alt="" />
          ) : (
            <span className="avatar-fallback">
              {name.slice(0, 1).toUpperCase()}
            </span>
          )}
          <span className="user-copy">
            <strong>{name}</strong>
            <small>{email}</small>
          </span>
        </button>
        {/* Photo picker, opened from Settings → Change photo */}
        <input
          accept="image/png,image/jpeg,image/webp"
          className="visually-hidden"
          onChange={onAvatarChange}
          ref={fileInput}
          type="file"
        />
      </div>
    </aside>
  );
}

function Icon({ active, item }: { active: boolean; item: string }) {
  // Each nav item has a white icon and a gold "-active" variant.
  return (
    <img
      alt=""
      className="sidebar-icon"
      src={asset(
        `sidebar-icons/${item.toLowerCase()}${active ? "-active" : ""}.svg`,
      )}
    />
  );
}
