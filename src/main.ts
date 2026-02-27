import Reveal from "reveal.js";
import "reveal.js/dist/reveal.css";
import "reveal.js/dist/theme/black.css";

const deck = new Reveal({
  hash: true,
  transition: "slide",
  mouseWheel: true,
});

deck.initialize();
