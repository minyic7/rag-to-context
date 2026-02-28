import "./rag-animation.css";
import type Reveal from "reveal.js";

export function initRagAnimation(deck: InstanceType<typeof Reveal>): void {
  const toggleAnimation = (slide: Element, active: boolean) => {
    const scene = slide.querySelector<SVGElement>(".rag-scene");
    if (!scene) return;

    if (active) {
      // For one-shot animations (fill-mode: forwards), setting animation: none
      // then clearing it forces the browser to restart from 0% on re-entry.
      // Infinite animations have no .tl-node elements so this is a no-op for them.
      scene.querySelectorAll<SVGElement>(".tl-node").forEach(node => {
        node.style.animation = "none";
      });
      scene.classList.remove("animate");
      void (scene as unknown as HTMLElement).offsetWidth; // reflow 1: apply animation:none
      scene.querySelectorAll<SVGElement>(".tl-node").forEach(node => {
        node.style.animation = "";
      });
      void (scene as unknown as HTMLElement).offsetWidth; // reflow 2: register fresh CSS animation
      scene.classList.add("animate");
    } else {
      scene.classList.remove("animate");
    }
  };

  // @ts-expect-error — reveal.js event typing is loose
  deck.on("slidechanged", (event: { currentSlide: Element; previousSlide: Element }) => {
    if (event.previousSlide) toggleAnimation(event.previousSlide, false);
    if (event.currentSlide)  toggleAnimation(event.currentSlide,  true);
  });

  // Handle case where the animation slide is the first slide shown
  const currentSlide = deck.getCurrentSlide?.();
  if (currentSlide) toggleAnimation(currentSlide, true);
}
