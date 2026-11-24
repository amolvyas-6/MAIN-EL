import "./styles.css";

export const metadata = {
  title: "NeRF2Mesh Studio",
  description: "Reconstruct a 3D model from photos and compare Instant-NGP GPU paths.",
};

export default function Layout({ children }) {
  return <html lang="en"><body>{children}</body></html>;
}
