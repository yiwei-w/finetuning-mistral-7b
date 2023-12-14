const katex = require('katex');
const fs = require('fs');
const path = require('path');
const markdownIt = require('markdown-it');
const markdownItKatex = require('@iktakahiro/markdown-it-katex');

// Initialize markdown-it with the KaTeX plugin
const md = markdownIt().use(markdownItKatex);

// Directory containing markdown files
const markdownDirectory = './../complete_files_md';

// Function to render LaTeX and catch errors
const renderAndCatchErrors = (latex, file, errors) => {
  try {
    katex.renderToString(latex);
  } catch (error) {
    errors.push({ file, latex, errorMessage: error.message });
  }
};

// Function to process a single markdown file
const processMarkdownFile = (filePath, errors) => {
  const content = fs.readFileSync(filePath, 'utf8');
  const tokens = md.parse(content, {});

  // Extract and render LaTeX snippets, and catch errors
  tokens.forEach(token => {
    if (token.type === 'inline' || token.type === 'math_block') {
      renderAndCatchErrors(token.content, filePath, errors);
    }
  });
};

// Function to process all markdown files and output errors
// const processAllMarkdownFiles = (directory) => {
//   const errors = [];
//   const files = fs.readdirSync(directory);

//   files.forEach(file => {
//     if (path.extname(file) === '.md') {
//       processMarkdownFile(path.join(directory, file), errors);
//     }
//   });

//   // Output errors to a file
//   const errorReport = errors.map(error => `File: ${error.file}\nLaTeX: ${error.latex}\nError: ${error.errorMessage}\n`).join('\n');
//   fs.writeFileSync('./katex-errors.txt', errorReport);
// };


// Function to process all markdown files and output errors
const processAllMarkdownFiles = (directory) => {
  const errors = [];
  const files = fs.readdirSync(directory);

  files.forEach(file => {
    if (path.extname(file) === '.md') {
      processMarkdownFile(path.join(directory, file), errors);
    }
  });

  // Output only error messages to a file
  const errorReport = errors.map(error => `File: ${error.file}\nError: ${error.errorMessage}\n`).join('\n');
  fs.writeFileSync('./katex-errors.txt', errorReport);
};

// Process all markdown files in the directory
processAllMarkdownFiles(markdownDirectory);