-- DO NOT TOUCH, ONLY TO KEEP TRACK OF QUERIES RAN IN SUPABASE
-- Remember which Google Calendar event a meeting created so deleting it in Iris
-- can also remove it from the user's Google Calendar.
alter table public.meetings add column if not exists google_event_id text;

notify pgrst, 'reload schema';
