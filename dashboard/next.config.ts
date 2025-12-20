import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Set correct root to avoid parent lockfile detection issues
  turbopack: {
    root: path.resolve(__dirname, ".."),
  },
};

export default nextConfig;
