export default function Loader({ label = "Loading" }) {
  return (
    <span className="inline-flex items-center gap-2 font-mono text-sm font-black uppercase">
      <span className="h-4 w-4 animate-spin border-2 border-black border-t-transparent dark:border-[#f8f4df] dark:border-t-transparent" />
      {label}
    </span>
  );
}
