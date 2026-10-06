import { useEffect, useState, type ChangeEvent } from "react";
const nav = [
  "Home",
  "Meetings",
  "Calendar",
  "Translation",
  "Settings",
];
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
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem("iris-sidebar-collapsed") === "true",
  );
  useEffect(
    () => localStorage.setItem("iris-sidebar-collapsed", String(collapsed)),
    [collapsed],
  );
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed((value) => !value)}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        type="button"
      >
        {collapsed ? "›" : "‹"}
      </button>
      <a className="wordmark" href="#home">
        <img src="/iris-mark.png" alt="" />
        <span>Iris</span>
      </a>
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
      src={`/sidebar-icons/${item.toLowerCase()}${active ? "-active" : ""}.svg`}
    />
  );
}
