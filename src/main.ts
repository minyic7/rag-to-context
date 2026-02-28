import Reveal from "reveal.js";
import "reveal.js/dist/reveal.css";
import "reveal.js/dist/theme/black.css";
import { initRagAnimation } from "./rag-animation";
import { initNaiveRagDemo } from "./naive-rag-demo";

const TRANSITION = "slide";
const TRANSITION_MS = 400;   // matches reveal.js default slide transition duration
const NAV_COOLDOWN_MS = 1000; // macOS momentum can last ~1s; block until it decays
const WHEEL_THRESHOLD = 50;  // ignore weak events (momentum tail has deltaY < 30)

const deck = new Reveal({
  hash: true,
  transition: TRANSITION,
  mouseWheel: false, // we handle wheel ourselves for interruptible navigation
});

deck.initialize().then(() => {
  initRagAnimation(deck);
  initNaiveRagDemo();
  let transitioning = false;
  let locked = false;
  let transitionTimer: ReturnType<typeof setTimeout> | null = null;
  let lockTimer: ReturnType<typeof setTimeout> | null = null;

  const navigate = (direction: "next" | "prev") => {
    if (locked) return; // cooldown: swallow momentum / rapid-fire events

    const go = () => (direction === "next" ? deck.next() : deck.prev());

    if (transitioning) {
      // Interrupt: snap in-flight animation to end state, navigate instantly
      deck.configure({ transition: "none" });
      go();
      deck.configure({ transition: TRANSITION });
      if (transitionTimer) clearTimeout(transitionTimer);
      transitioning = false;
    } else {
      transitioning = true;
      if (transitionTimer) clearTimeout(transitionTimer);
      transitionTimer = setTimeout(() => {
        transitioning = false;
      }, TRANSITION_MS);
      go();
    }

    // Lock out further navigation until cooldown expires
    locked = true;
    if (lockTimer) clearTimeout(lockTimer);
    lockTimer = setTimeout(() => {
      locked = false;
    }, NAV_COOLDOWN_MS);
  };

  const el = document.querySelector(".reveal") as HTMLElement;
  el.addEventListener(
    "wheel",
    (e: WheelEvent) => {
      e.preventDefault();
      if (Math.abs(e.deltaY) < WHEEL_THRESHOLD) return;
      navigate(e.deltaY > 0 ? "next" : "prev");
    },
    { passive: false }
  );
});
