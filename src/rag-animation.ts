import "./rag-animation.css";
import type Reveal from "reveal.js";

export function initRagAnimation(deck: InstanceType<typeof Reveal>): void {
  const toggleAnimation = (slide: Element, active: boolean) => {
    const scene = slide.querySelector<SVGElement>(".rag-scene");
    if (!scene) return;

    if (active) {
      // Reset by removing class, forcing reflow, then re-adding
      scene.classList.remove("animate");
      void (scene as unknown as HTMLElement).offsetWidth;
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
