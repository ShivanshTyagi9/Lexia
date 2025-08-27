export async function answerQuestions({
  question,
  documentContent
}: {
  question: string;
  documentContent: string;
}) {
  const res = await fetch("http://localhost:8000/query", {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_content: documentContent }),
  });

  if (!res.ok) {
    throw new Error("Failed to get response from backend");
  }

  return await res.json(); // should be { answer: "..." }
}
