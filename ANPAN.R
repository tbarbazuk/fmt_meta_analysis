library(anpan)
library(ape)
library(dplyr)
library(tidyr)
library(ggtree)
library(ggtreeExtra)
library(ggnewscale)
library(ggplot2)

# Force dplyr verbs to win over Bioconductor S4 generics for the rest of the script
select <- dplyr::select
filter <- dplyr::filter
rename <- dplyr::rename
mutate <- dplyr::mutate

## ============================================================
# 1. Load data
# ============================================================
tree <- read.tree("/Users/thomasbarbazuk/CIBIO/meta-analysis-phd/pruned_tree.nwk")

traits <- read.csv(
  #"/Users/thomasbarbazuk/CIBIO/meta-analysis-phd/SGB_2_Acquisition.csv",
  "/Users/thomasbarbazuk/CIBIO/meta-analysis-phd/anpan_for_submission.csv",
  header = TRUE, stringsAsFactors = FALSE
)

traits <- traits |>
  rename(sample_id = sgb, outcome = residual) |>
  mutate(
    sample_id = as.character(sample_id),
    sample_id = sub("^t__SGB", "", sample_id),
    sample_id = sub("_group$", "", sample_id)
  )

stopifnot(all(tree$tip.label %in% traits$sample_id))
stopifnot(all(traits$sample_id %in% tree$tip.label))

# ============================================================
# 2. Fit PGLMM (unchanged)
# ============================================================
result <- anpan_pglmm(
  meta_file = traits,
  tree_file = tree,
  outcome = "outcome",
  covariates = NULL,
  family = "gaussian",
  bug_name = "sgb_acquisition",
  reg_noise = TRUE,
  loo_comparison = TRUE,
  run_diagnostics = FALSE,
  show_plot_tree = FALSE,
  show_post = FALSE,
  refresh = 500
)
result$loo$comparison

# ============================================================
# 3. Extract posterior phylo effects per tip
# ============================================================
draw_df <- result$pglmm_fit$draws(format = "df")
draw_df <- as.data.frame(draw_df)   # <-- drop draws_df/S4-triggering class

phylo_cols <- grep("^phylo_effect\\[", names(draw_df), value = TRUE)

post_long <- draw_df %>%
  dplyr::select(all_of(phylo_cols)) %>%     # explicit namespace, belt-and-suspenders
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  mutate(id = as.integer(gsub("phylo_effect\\[|\\]", "", variable))) %>%
  group_by(id) %>%
  summarise(mean = mean(value),
            `5%`  = quantile(value, 0.05),
            `95%` = quantile(value, 0.95),
            .groups = "drop")
tip_map <- result$model_input$sample_id
post_long <- post_long %>% mutate(label = tip_map[id])

effect_df <- post_long %>% select(label, mean)

# ============================================================
# 4. Build annotation table: outcome, phylum, median donor abundance
#    (all keyed on 'label' = tip label = sample_id)
# ============================================================
trait_df <- traits %>%
  rename(label = sample_id) %>%
  select(label, outcome, phylum, median_donor_abundance, log10_abundance) %>%
  mutate(
    phylum = factor(phylum)
  )

n_outcome_vals <- n_distinct(trait_df$outcome)
outcome_is_categorical <- n_outcome_vals <= 10

message("outcome has ", n_outcome_vals, " unique values -> treating as ",
        if (outcome_is_categorical) "CATEGORICAL" else "CONTINUOUS")

if (outcome_is_categorical) {
  trait_df <- trait_df %>% mutate(outcome = factor(outcome))
} else {
  trait_df <- trait_df %>% mutate(outcome = as.numeric(outcome))
}

# ============================================================
# 5. Color palettes (revised phylum palette)
# ============================================================
if (outcome_is_categorical) {
  outcome_colors <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
                       "#E6AB02", "#A6761D", "#666666")
  outcome_pal <- setNames(
    outcome_colors[seq_len(nlevels(trait_df$outcome))],
    levels(trait_df$outcome)
  )
} else {
  # Residuals can be negative -- use a symmetric range around 0 so the
  # diverging scale's midpoint is meaningful regardless of skew.
  outcome_abs_max <- max(abs(trait_df$outcome), na.rm = TRUE)
  outcome_limits <- c(-outcome_abs_max, outcome_abs_max)
}

# ---- Phylum palette ----
phylum_levels <- levels(trait_df$phylum)
n_phyla <- length(phylum_levels)

# Hand-picked, maximally distinct hues for the 4 dominant phyla in gut
# microbiome data, plus a hard-to-confuse color for "unclassified".
fixed_phylum_colors <- c(
  "Bacteroidota"     = "#E31A1C",
  "Pseudomonadota"   = "#1F78B4",
  "Firmicutes"       = "#33A02C",
  "Bacillota"        = "#33A02C",
  "Actinomycetota"   = "#FF7F00",
  "Actinobacteria"   = "#FF7F00",
  "Unclassified"     = "#4D4D4D",
  "unclassified"     = "#4D4D4D",
  "Unknown"          = "#4D4D4D",
  "unknown"          = "#4D4D4D"
)

matched <- intersect(phylum_levels, names(fixed_phylum_colors))
unmatched <- setdiff(phylum_levels, names(fixed_phylum_colors))

extra_colors <- c("#B15928", "#6A3D9A", "#FDBF6F", "#CAB2D6",
                   "#FFFF99", "#A6CEE3", "#B2DF8A", "#FB9A99")
extra_pal <- setNames(
  extra_colors[((seq_along(unmatched) - 1) %% length(extra_colors)) + 1],
  unmatched
)

phylum_pal <- c(fixed_phylum_colors[matched], extra_pal)
phylum_pal <- phylum_pal[phylum_levels]

# ---- Phylum shapes ----
phylum_shapes_pool <- c(21, 22, 23, 24, 25, 21, 22, 23)
is_unclassified <- grepl("unclass|unknown", phylum_levels, ignore.case = TRUE)
phylum_shape_pal <- setNames(rep(NA_integer_, n_phyla), phylum_levels)
phylum_shape_pal[!is_unclassified] <- phylum_shapes_pool[
  ((seq_len(sum(!is_unclassified)) - 1) %% length(phylum_shapes_pool)) + 1
]
phylum_shape_pal[is_unclassified] <- 8

# ============================================================
# 6. Base tree (clean, no tip labels, thin branches)
# ============================================================
p <- ggtree(tree, layout = "circular", size = 0.3, color = "grey30")

tip_coords <- p$data %>% filter(isTip) %>% select(label, x, y)
tip_df <- tip_coords %>% left_join(trait_df, by = "label")

# ============================================================
# 7. Layer the rings + tip shapes
# ============================================================
p2 <- p +

  # ---- Tip shapes: Phylum, right at the leaf ends ----
  geom_point(
    data = tip_df,
    mapping = aes(x = x, y = y, fill = phylum, shape = phylum),
    size = 2, color = "black", stroke = 0.2,
    inherit.aes = FALSE
  ) +
  scale_fill_manual(values = phylum_pal, name = "Phylum") +
  scale_shape_manual(values = phylum_shape_pal, name = "Phylum") +
  new_scale_fill() +

  # ---- Ring 1: Acquisition (residual, adjusted for abundance) ----
  geom_fruit(
    data = trait_df, geom = geom_tile,
    mapping = aes(y = label, fill = outcome),
    width = 0.18, offset = 0.10, color = "white", pheight = 0.3
  ) +
  { if (outcome_is_categorical) {
      scale_fill_manual(values = outcome_pal, name = "Acquisition")
    } else {
      scale_fill_gradient2(
        low = "#4575B4", mid = "white", high = "#D73027",
        midpoint = 0,
        limits = outcome_limits,
        name = "Acquisition (residual)",
        na.value = "grey90"
      )
    }
  } +
  new_scale_fill() +

    # ---- Ring 2: ANPAN phylogenetic effect ----
  geom_fruit(
    data = effect_df, geom = geom_tile,
    mapping = aes(y = label, fill = mean),
    width = 0.18, offset = 0.03, color = "white"
  ) +
  scale_fill_gradient2(
    low = "#762A83", mid = "white", high = "#1B7837",
    midpoint = 0, name = "ANPAN Phylogenetic effect"
  ) +
  new_scale_fill() +

  # ---- Ring 3: Median donor abundance (log10 scale) ----
  geom_fruit(
    data = trait_df, geom = geom_tile,
    mapping = aes(y = label, fill = log10_abundance),
    width = 0.18, offset = 0.03, color = "white"
  ) +
  scale_fill_viridis_c(
    name = "Median Donor Abundance (log10)",
    option = "magma",
    na.value = "grey90"
  ) +

  # ---- Theme / cleanup ----
  theme(
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text  = element_text(size = 8),
    legend.key.size = unit(0.4, "cm"),
    plot.margin = margin(10, 10, 10, 10)
  )

p2

ggsave("/Users/thomasbarbazuk/Downloads/anpan_tree.svg", p2, width = 12, height = 12)
