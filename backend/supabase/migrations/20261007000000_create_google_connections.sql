-- DO NOT TOUCH, ONLY TO KEEP TRACK OF QUERIES RAN IN SUPABASE
-- Long-lived Google refresh tokens, one per Iris user. Only the
-- google-calendar-token Edge Function (service role) can read or write this
-- table; the desktop app never sees refresh tokens after handing them over.
create table if not exists public.google_connections (
  user_id uuid primary key references auth.users (id) on delete cascade,
  refresh_token text not null,
  updated_at timestamptz not null default now()
);

alter table public.google_connections enable row level security;
-- No RLS policies on purpose: signed-in users cannot query this table directly.
revoke all on public.google_connections from anon, authenticated;
grant select, insert, update, delete on public.google_connections to service_role;

notify pgrst, 'reload schema';
