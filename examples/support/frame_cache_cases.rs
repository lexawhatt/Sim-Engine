//! Case filtering never changes a workload's content, quality or source counts.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct Selection {
    pub(super) quick: bool,
    pub(super) labels: Option<usize>,
    pub(super) trials: usize,
    pub(super) upload_staging_bytes: Option<usize>,
    content: Option<Content>,
    cached: Option<bool>,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            quick: false,
            labels: None,
            trials: 1,
            upload_staging_bytes: None,
            content: None,
            cached: None,
        }
    }
}

impl Case {
    pub(super) fn cache_budget(self, upload_staging_bytes: Option<usize>) -> FrameCacheBudget {
        let budget = if self.cached {
            FrameCacheBudget::default()
        } else {
            FrameCacheBudget::new(0, 0, 0, 0)
        };
        upload_staging_bytes.map_or(budget, |bytes| budget.with_upload_staging_bytes(bytes))
    }
}

impl Selection {
    pub(super) fn set_content(&mut self, name: &str) -> Result<()> {
        self.content = Some(match name {
            "world" => Content::World,
            "world_screen" => Content::WorldScreen,
            "mixed64_images128rects" => Content::Mixed64,
            "unchanged" => Content::Labels,
            "recolored" => Content::Recolored,
            "moved" => Content::Moved,
            "mixed" => Content::Mixed,
            _ => return Err(format!("unknown benchmark case {name}").into()),
        });
        Ok(())
    }

    pub(super) fn set_cache(&mut self, name: &str) -> Result<()> {
        self.cached = match name {
            "on" => Some(true),
            "off" => Some(false),
            "both" => None,
            _ => return Err("--cache must be on, off or both".into()),
        };
        Ok(())
    }

    pub(super) fn cases(self) -> Result<Vec<Case>> {
        if !(1..=100).contains(&self.trials) {
            return Err("--trials must be in 1..=100".into());
        }
        let mut cases = Vec::new();
        for content in [Content::World, Content::WorldScreen, Content::Mixed64] {
            for cached in [false, true] {
                cases.push(Case {
                    content,
                    labels: 0,
                    cached,
                });
            }
        }
        let counts: &[usize] = if self.quick { &[1] } else { &[1, 100, 1000] };
        for &labels in counts {
            for content in [
                Content::Labels,
                Content::Recolored,
                Content::Moved,
                Content::Mixed,
            ] {
                for cached in [false, true] {
                    cases.push(Case {
                        content,
                        labels,
                        cached,
                    });
                }
            }
        }
        cases.retain(|case| {
            self.content.is_none_or(|content| content == case.content)
                && self.labels.is_none_or(|labels| labels == case.labels)
                && self.cached.is_none_or(|cached| cached == case.cached)
        });
        if cases.is_empty() {
            return Err(
                "benchmark selection matches no cases (check --quick/--case/--labels)".into(),
            );
        }
        let mut trials = Vec::with_capacity(cases.len() * self.trials);
        for trial in 0..self.trials {
            if trial % 2 == 0 {
                trials.extend_from_slice(&cases);
            } else {
                trials.extend(cases.iter().rev().copied());
            }
        }
        Ok(trials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cases_preserve_all_original_workloads_and_order() {
        let cases = Selection::default().cases().unwrap();
        assert_eq!(cases.len(), 30);
        for pair in cases.chunks_exact(2) {
            assert!(!pair[0].cached);
            assert!(pair[1].cached);
            assert_eq!(pair[0].content, pair[1].content);
            assert_eq!(expected_counts(pair[0]), expected_counts(pair[1]));
        }
        assert_eq!(cases[0].content, Content::World);
        assert_eq!(cases[29].content, Content::Mixed);
        assert_eq!(cases[29].labels, 1000);
    }

    #[test]
    fn focused_trials_alternate_order_without_changing_content() {
        let mut selection = Selection {
            labels: Some(1000),
            trials: 3,
            ..Selection::default()
        };
        selection.set_content("mixed").unwrap();
        let cases = selection.cases().unwrap();
        assert_eq!(
            cases.iter().map(|case| case.cached).collect::<Vec<_>>(),
            [false, true, true, false, false, true]
        );
        assert!(
            cases
                .iter()
                .all(|case| expected_counts(*case) == (1001, 1000, 1000, 3100, 3001))
        );
        selection.set_cache("on").unwrap();
        assert_eq!(selection.cases().unwrap().len(), 3);
        assert!(
            selection
                .cases()
                .unwrap()
                .iter()
                .all(|case| case.cache_budget(None) == FrameCacheBudget::default())
        );
        for case in selection.cases().unwrap() {
            let direct = case.cache_budget(Some(0));
            assert_eq!(direct.max_upload_staging_bytes(), 0);
            assert_eq!(
                direct.max_bindings(),
                FrameCacheBudget::default().max_bindings()
            );
            assert_eq!(
                direct.max_uniform_bytes(),
                FrameCacheBudget::default().max_uniform_bytes()
            );
        }
    }

    #[test]
    fn invalid_or_empty_selection_is_rejected_before_window_creation() {
        for trials in [0, 101, usize::MAX] {
            assert!(
                Selection {
                    trials,
                    ..Selection::default()
                }
                .cases()
                .is_err()
            );
        }
        for labels in [2, usize::MAX] {
            assert!(
                Selection {
                    labels: Some(labels),
                    ..Selection::default()
                }
                .cases()
                .is_err()
            );
        }
        assert!(
            Selection {
                quick: true,
                labels: Some(1000),
                ..Selection::default()
            }
            .cases()
            .is_err()
        );
        assert!(Selection::default().set_content("unknown").is_err());
        assert!(Selection::default().set_cache("unknown").is_err());
    }
}
