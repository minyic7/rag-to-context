import Reveal from "reveal.js";
import "reveal.js/dist/reveal.css";
import "reveal.js/dist/theme/black.css";

const TRANSITION = "slide";
const TRANSITION_MS = 400; // matches reveal.js default slide transition duration

const deck = new Reveal({
  hash: true,
  transition: TRANSITION,
  mouseWheel: false, // we handle wheel ourselves for interruptible navigation
});

deck.initialize().then(() => {
  let transitioning = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const navigate = (direction: "next" | "prev") => {
    const go = () => (direction === "next" ? deck.next() : deck.prev());

    if (transitioning) {
      // Interrupt: snap the in-flight animation to its end state, then navigate instantly
      deck.configure({ transition: "none" });
      go();
      deck.configure({ transition: TRANSITION });
      if (timer) clearTimeout(timer);
      transitioning = false;
      return;
    }

    transitioning = true;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      transitioning = false;
    }, TRANSITION_MS);
    go();
  };

  const el = document.querySelector(".reveal") as HTMLElement;
  el.addEventListener(
    "wheel",
    (e: WheelEvent) => {
      e.preventDefault();
      navigate(e.deltaY > 0 ? "next" : "prev");
    },
    { passive: false }
  );
});
