export default function VerificationBadge({ score = 0 }) {
  const value = Number(score) || 0;
  const tone =
    value >= 70
      ? "bg-[#7cf7c7]"
      : value >= 40
        ? "bg-[#ffdf38]"
        : "bg-[#ff7a7a]";

  return (
    <div className={`inline-flex border-2 border-black px-3 py-2 font-mono font-black uppercase text-black shadow-brutal dark:border-[#f8f4df] dark:shadow-brutal-dark ${tone}`}>
      Verification Score: {value.toFixed(value % 1 ? 1 : 0)}
    </div>
  );
}
