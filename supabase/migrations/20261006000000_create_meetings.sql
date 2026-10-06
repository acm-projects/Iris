-- Meetings shown on the Iris dashboard. Each row belongs to the user who created it.
create table if not exists public.meetings (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  title text not null check (char_length(title) between 1 and 200),
  starts_at timestamptz not null,
  ends_at timestamptz not null,
  meeting_url text check (meeting_url is null or meeting_url like 'https://meet.google.com/%'),
  accessibility_mode text not null default 'Captions + Sign translation',
  color text not null default 'blue',
  description text,
  created_at timestamptz not null default now(),
  constraint meetings_ends_after_start check (ends_at > starts_at)
);

create index if not exists meetings_owner_starts_idx
  on public.meetings (owner_id, starts_at);

-- Signed-in users may use the table at all; RLS below limits them to their own rows.
-- Anonymous (signed-out) requests get no access.
grant select, insert, update, delete on public.meetings to authenticated;
revoke all on public.meetings from anon;

-- Row Level Security: users can only see and change their own meetings.
alter table public.meetings enable row level security;

drop policy if exists "Users read own meetings" on public.meetings;
create policy "Users read own meetings" on public.meetings
  for select to authenticated using (owner_id = auth.uid());

drop policy if exists "Users create own meetings" on public.meetings;
create policy "Users create own meetings" on public.meetings
  for insert to authenticated with check (owner_id = auth.uid());

drop policy if exists "Users update own meetings" on public.meetings;
create policy "Users update own meetings" on public.meetings
  for update to authenticated
  using (owner_id = auth.uid()) with check (owner_id = auth.uid());

drop policy if exists "Users delete own meetings" on public.meetings;
create policy "Users delete own meetings" on public.meetings
  for delete to authenticated using (owner_id = auth.uid());

-- Make the new table visible to the API immediately.
notify pgrst, 'reload schema';
