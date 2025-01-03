export interface Resource {
  title: string;
  description: string;
  url: string;
}

export interface CodeExample {
  language: string;
  code: string;
  description: string;
}

export interface NodeContent {
  title: string;
  description: string;
  concepts: string[];
  examples: CodeExample[];
  resources: Resource[];
  prerequisites?: string[];
  relatedTopics?: string[];
}

export type NodeContentMap = {
  [nodeType: string]: {
    [nodeId: string]: NodeContent;
  };
};