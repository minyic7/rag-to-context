declare module "reveal.js" {
  interface RevealOptions {
    hash?: boolean;
    transition?: string;
    [key: string]: unknown;
  }

  export default class Reveal {
    constructor(options?: RevealOptions);
    initialize(): Promise<void>;
  }
}

declare module "reveal.js/dist/reveal.css";
declare module "reveal.js/dist/theme/black.css";
