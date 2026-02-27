declare module "reveal.js" {
  interface RevealOptions {
    hash?: boolean;
    transition?: string;
    transitionSpeed?: "default" | "fast" | "slow";
    mouseWheel?: boolean;
    [key: string]: unknown;
  }

  export default class Reveal {
    constructor(options?: RevealOptions);
    initialize(): Promise<void>;
    configure(options: Partial<RevealOptions>): void;
    next(): void;
    prev(): void;
    on(event: string, callback: (event: unknown) => void): void;
    getCurrentSlide(): Element | null;
  }
}

declare module "reveal.js/dist/reveal.css";
declare module "reveal.js/dist/theme/black.css";
