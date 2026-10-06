import type { ChangeEvent } from "react";
const nav = [
  "Home",
  "Meetings",
  "Calendar",
  "Translation",
  "Analytics",
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
  return (
    <aside className="sidebar">
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
            type="button"
          >
            <Icon active={active === item} item={item} />
            {item}
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
          <span>
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
        <button className="sidebar-sign-out" onClick={onSignOut} type="button">
          ↪ Sign out
        </button>
      </div>
    </aside>
  );
}
function Icon({ active, item }: { active: boolean; item: string }) {
  const icon =
    item === "Home"
      ? "home"
      : item === "Meetings"
        ? "meetings"
        : item === "Calendar"
          ? "calendar"
          : item === "Settings"
            ? "settings"
            : null;
  return icon ? (
    <img
      alt=""
      className="sidebar-icon"
      src={`/sidebar-icons/${icon}${active ? "-active" : ""}.svg`}
    />
  ) : (
    <span className="sidebar-fallback">
      {item === "Translation" ? "⌁" : "◌"}
    </span>
  );
}
