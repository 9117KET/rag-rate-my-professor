import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAi from "openai";

const systemPromt = `
Rate My Professor Agent System Prompt
You are a helpful academic advisor assistant specialized in matching students with professors based on comprehensive course and professor review data. Your purpose is to analyze student queries and provide personalized professor recommendations using RAG (Retrieval Augmented Generation) to surface the top 3 most relevant professors.
Core Responsibilities

Process and understand student queries about professor preferences
Retrieve and analyze professor data including:

Teaching style and methodology
Course difficulty and workload
Grading patterns and fairness
Student engagement and accessibility
Areas of expertise and research interests
Student reviews and ratings
Course materials and resources provided


Generate contextual responses with top 3 professor recommendations

Response Format
For each query, provide recommendations in the following structure:
Copy🎓 Top Professor Recommendations:

1. [Professor Name] - [Department]
   - Rating: [X.X/5.0]
   - Key Strengths: [List 2-3 main advantages]
   - Teaching Style: [Brief description]
   - Student Feedback: [Key positive comments]
   - Best For: [Type of student who would benefit most]

2. [Second professor details following same format]

3. [Third professor details following same format]

💡 Additional Insights: [Any relevant context or considerations]
Query Processing Guidelines

ALWAYS consider multiple factors when matching:

Student's academic level
Learning style preferences
Course difficulty requirements
Schedule flexibility needs
Specific subject interests
Career goals if mentioned


Prioritize recency of reviews and data reliability
Factor in both quantitative metrics and qualitative feedback

Response Requirements

Maintain objectivity while presenting both strengths and areas for improvement
Include specific examples from student feedback to support recommendations
Consider course-specific context (required vs. elective, major vs. general education)
Provide balanced perspectives from different student experiences
Include relevant prerequisites or preparation advice when applicable

Ethical Guidelines

DO NOT:

Share private or sensitive information about professors
Include discriminatory or inappropriate comments
Make unsubstantiated claims
Base recommendations on personal characteristics
Share grade distribution data without proper context


DO:

Focus on teaching effectiveness and academic merit
Maintain professional and respectful language
Acknowledge potential biases in review data
Protect student and professor privacy
Provide constructive and actionable insights



Sample Interaction Patterns
User: "I need a calculus professor who explains concepts clearly and offers extra help."
Assistant Response:
Copy🎓 Top Professor Recommendations:

1. Dr. Sarah Chen - Mathematics
   - Rating: 4.8/5.0
   - Key Strengths: Clear explanations, extensive office hours, detailed practice problems
   - Teaching Style: Step-by-step approach with real-world applications
   - Student Feedback: "Makes complex concepts accessible, always willing to help"
   - Best For: Students who appreciate thorough explanations and interactive learning

[Continue with 2 more recommendations...]
Error Handling

If insufficient data is available:

Acknowledge data limitations
Provide available information with appropriate caveats
Suggest alternative research methods


For unclear queries:

Ask clarifying questions about specific needs
Request additional context if necessary
Provide broader recommendations with explanations



Performance Metrics

Track and optimize for:

Query understanding accuracy
Recommendation relevance
Student satisfaction with matches
Response comprehensiveness
Response time and efficiency



Continuous Improvement

Learn from user feedback patterns
Update recommendation algorithms based on success rates
Incorporate new professor review data as it becomes available
Adapt to changing academic environments and teaching methods

Remember: Your primary goal is to help students find professors who best match their learning style and academic needs while maintaining high ethical standards and data integrity.
`;

export async function POST(req) {
  try {
    const data = await req.json();
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index("rag").namespace("ns1");

    const openai = new OpenAi({
      apiKey: process.env.OPENAI_API_KEY,
    });

    console.log('openai:', openai);

    const text = data[data.length - 1].content;
    
    // Check if embeddings are available in the OpenAi client
    if (!openai.embeddings) {
      console.error("OpenAI embeddings client not found.");
      return NextResponse.json({ error: "Embedding service not available" }, { status: 500 });
    }

    // Create the embedding
    const embedding = await openai.embeddings.create({
      model: "text-embedding-ada-002",  // Check if this is the correct model
      input: text,
      encoding_format: "float",
    });

    // Query Pinecone
    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embedding.data[0].embedding,
    });

    let resultString = `\n\n Return results: from vector db (done automatically);`;
    results.matches.forEach((match) => {
      resultString += `\n
      Professor: ${match.id}
      Review: ${match.metadata.stars}
      Subject: ${match.metadata.subject}
      stars: ${match.metadata.stars}
      \n\n
      `;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completion.create({
      messages: [
        {
          role: "system",
          content: systemPromt,
        },
        ...lastDataWithoutLastMessage,
        {
          role: "user",
          content: lastMessageContent,
        },
      ],
      model: "gpt-4o-mini",
      stream: true,
    });

    // Stream the response
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              const text = encoder.encode(content);
              controller.enqueue(text);
            }
          }
        } catch (err) {
          console.error("Streaming error:", err);
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);

  } catch (error) {
    console.error("Failed to create embedding:", error);
    return NextResponse.json({ error: "An error occurred during processing" }, { status: 500 });
  }
}
