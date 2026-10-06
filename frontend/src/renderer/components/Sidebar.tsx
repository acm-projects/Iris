// Sidebar shown on every signed-in page.
import { useEffect, useState, type ChangeEvent } from "react";
import { asset } from "../utils/assets";
// Pages in the order they appear. Each name maps to an icon in
// assets/sidebar-icons/<name>.svg (and <name>-active.svg when selected).
const nav = ["Home", "Calendar", "Translation", "Settings"];
/** Left navigation: page links, profile photo/name, and sign out. */
export default function Sidebar({
  active,
  avatarUrl,
  fileInput,
  name,
  onAvatarChange,
  onNavigate,
  onSignOut,
}: {
  active: string;
  avatarUrl: string | null;
  fileInput: React.RefObject<HTMLInputElement | null>;
  name: string;
  onAvatarChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onNavigate: (page: string) => void;
  onSignOut: () => void;
}) {
  // Collapsed (icons only) or expanded; remembered between launches.
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem("iris-sidebar-collapsed") === "true",
  );
  useEffect(
    () => localStorage.setItem("iris-sidebar-collapsed", String(collapsed)),
    [collapsed],
  );
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      {/* Collapse / expand toggle on the sidebar's edge */}
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed((value) => !value)}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        type="button"
      >
        {collapsed ? "›" : "‹"}
      </button>
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
            title={collapsed ? item : undefined}
            type="button"
          >
            <Icon active={active === item} item={item} />
            <em>{item}</em>
          </button>
        ))}
      </nav>
      {/* Profile: clicking the photo opens a file picker to change it */}
      <div className="sidebar-footer">
        <button
          className="user-chip"
          onClick={() => fileInput.current?.click()}
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
            <small>Change photo</small>
          </span>
        </button>
        <input
          accept="image/png,image/jpeg,image/webp"
          className="visually-hidden"
          onChange={onAvatarChange}
          ref={fileInput}
          type="file"
        />
        {/* Sign out */}
        <button
          className="sidebar-sign-out"
          onClick={onSignOut}
          title="Sign out"
          type="button"
        >
          ↪ <em>Sign out</em>
        </button>
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
