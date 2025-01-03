import React from 'react';
import CodeBlock from '../../CodeBlock';

function GitHubTools() {
  return (
    <section id="github-tools">
      <h2>GitHub Tools</h2>
      <p>
        Advanced tools for GitHub integration, including repository management, workflow automation,
        and issue tracking.
      </p>

      <h3>Repository Management</h3>
      <CodeBlock
        language="typescript"
        code={`interface GitHubConfig {
  auth: {
    type: 'token' | 'oauth' | 'app';
    credentials: {
      token?: string;
      clientId?: string;
      clientSecret?: string;
      privateKey?: string;
    };
  };
  repository: {
    owner: string;
    name: string;
    branch: string;
    webhook?: {
      secret: string;
      events: string[];
    };
  };
  workflow: {
    actions: {
      enabled: boolean;
      permissions: string[];
      environment: Record<string, string>;
    };
    artifacts: {
      retention: number;
      maxSize: number;
    };
  };
}

class GitHubManager {
  async createPullRequest(
    title: string,
    branch: string,
    config: GitHubConfig
  ) {
    // Initialize GitHub client
    const client = await this.createClient(config.auth);
    
    // Create branch if doesn't exist
    await this.ensureBranch(client, branch);
    
    // Create commit with changes
    const commit = await this.createCommit(client, {
      branch,
      message: title,
      files: this.getChangedFiles()
    });
    
    // Create pull request
    return client.pulls.create({
      owner: config.repository.owner,
      repo: config.repository.name,
      title,
      head: branch,
      base: config.repository.branch,
      body: this.generatePRDescription(commit)
    });
  }

  async setupWorkflowAutomation(config: GitHubConfig) {
    // Configure GitHub Actions
    if (config.workflow.actions.enabled) {
      await this.configureActions(
        config.workflow.actions
      );
    }
    
    // Setup webhooks
    if (config.repository.webhook) {
      await this.createWebhook(
        config.repository,
        config.repository.webhook
      );
    }
    
    // Configure branch protection
    await this.configureBranchProtection({
      branch: config.repository.branch,
      requiredReviews: 1,
      requiredChecks: true
    });
  }
}`}
      />

      <h3>Issue Management</h3>
      <CodeBlock
        language="typescript"
        code={`interface IssueConfig {
  templates: {
    bug: string;
    feature: string;
    custom: Record<string, string>;
  };
  labels: {
    name: string;
    color: string;
    description?: string;
  }[];
  automation: {
    autoAssign: boolean;
    autoLabel: boolean;
    autoClose: {
      enabled: boolean;
      inactiveDays: number;
    };
  };
  projects: {
    enabled: boolean;
    defaultColumn: string;
    automation: {
      moveOnStatus: boolean;
      moveOnLabel: boolean;
    };
  };
}

class IssueManager {
  async createIssue(
    data: IssueData,
    config: IssueConfig
  ) {
    // Apply issue template
    const body = this.applyTemplate(
      data,
      config.templates
    );
    
    // Create issue
    const issue = await this.client.issues.create({
      ...data,
      body
    });
    
    // Apply automation rules
    if (config.automation.autoLabel) {
      await this.autoLabel(issue);
    }
    
    if (config.automation.autoAssign) {
      await this.autoAssign(issue);
    }
    
    // Add to project if enabled
    if (config.projects.enabled) {
      await this.addToProject(issue, config.projects);
    }
    
    return issue;
  }
}`}
      />
    </section>
  );
}

export default GitHubTools;