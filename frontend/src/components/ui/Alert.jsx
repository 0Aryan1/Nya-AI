export default function Alert({ title, children, tone = "info", action }) {
  const tones = {
    info: "bg-[#b4e7ff]",
    danger: "bg-[#ff7a7a]",
    warning: "bg-[#ffdf38]",
    success: "bg-[#7cf7c7]",
  };

  return (
    <div className={`border-2 border-black p-4 shadow-brutal dark:border-[#f8f4df] dark:shadow-brutal-dark ${tones[tone]}`}>
      {title ? <p className="font-mono text-sm font-black uppercase text-black">{title}</p> : null}
      <div className="mt-1 text-sm font-semibold text-black">{children}</div>
      {action ? <div className="mt-3">{action}</div> : null}
    </div>
  );
}
