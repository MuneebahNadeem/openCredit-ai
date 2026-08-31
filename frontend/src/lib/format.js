export function formatDateTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatScore(score) {
  if (score === null || score === undefined) return "—";
  return `${Math.round(score * 100)}`;
}

export function titleCase(value) {
  if (!value) return "";
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function prettyName(field) {
  const map = {
    instagram_followers: "Instagram followers",
    facebook_followers: "Facebook followers",
    reviews_rating: "Review rating",
    review_count: "Review count",
    units_sold: "Units sold",
    listing_count: "Active listings",
    self_reported_revenue: "Revenue (self-reported)",
    self_reported_orders: "Monthly orders (self-reported)",
    business_age: "Years in business",
    staff_count: "Staff",
    sales_channels: "Sales channels",
    product_types: "Product types",
    engagement_rate: "Engagement",
    recent_activity: "Recent activity",
  };
  return map[field] || titleCase(field);
}

export const SOURCE_TYPE_META = {
  social: { label: "Social", icon: "◎" },
  marketplace: { label: "Marketplace", icon: "▣" },
  review: { label: "Reviews", icon: "★" },
  government: { label: "Government", icon: "§" },
  web: { label: "Web", icon: "◇" },
};

export const RECOMMENDATION_META = {
  approve: {
    label: "Approve",
    description: "Both trustworthiness and business potential are high.",
  },
  approve_with_conditions: {
    label: "Approve with conditions",
    description: "Adequate on both axes; verify the flagged gaps first.",
  },
  decline: {
    label: "Decline",
    description: "Trustworthiness signals are weak relative to the risks.",
  },
  further_review: {
    label: "Further review",
    description: "Trust is adequate but business potential needs a closer look.",
  },
  insufficient_data: {
    label: "Insufficient data",
    description: "Not enough evidence to support either assessment.",
  },
};

export const TERMINAL_STATUSES = ["completed", "partial", "failed"];

export const POLL_MS = 1500;
