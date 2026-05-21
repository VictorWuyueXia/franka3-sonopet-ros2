#include <memory>
#include <string>

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include "fr3_sonopet_interfaces/action/execute_motion.hpp"
#include "fr3_sonopet_interfaces/action/preview_motion.hpp"

namespace fr3_sonopet_interfaces
{

class MotionControlPanel final : public rviz_common::Panel
{
public:
  explicit MotionControlPanel(QWidget * parent = nullptr)
  : rviz_common::Panel(parent)
  {
    auto * layout = new QVBoxLayout(this);
    auto * row = new QHBoxLayout();
    preview_button_ = new QPushButton("Preview", this);
    execute_button_ = new QPushButton("Execute", this);
    confirmation_ = new QLineEdit(this);
    status_ = new QLabel("Waiting for RViz node.", this);
    confirmation_->setMaxLength(1);
    confirmation_->setPlaceholderText("E");
    preview_button_->setEnabled(false);
    execute_button_->setEnabled(false);
    row->addWidget(preview_button_);
    row->addWidget(new QLabel("Confirm:", this));
    row->addWidget(confirmation_);
    row->addWidget(execute_button_);
    layout->addLayout(row);
    layout->addWidget(status_);
    connect(preview_button_, &QPushButton::clicked, this, [this]() {
      PreviewMotion::Goal goal;
      rclcpp_action::Client<PreviewMotion>::SendGoalOptions options;
      status_->setText("Preview requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Preview accepted." : "Preview rejected.");
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      preview_client_->async_send_goal(goal, options);
    });
    connect(execute_button_, &QPushButton::clicked, this, [this]() {
      if (confirmation_->text() != "E") {
        status_->setText("Type E to execute.");
        return;
      }
      ExecuteMotion::Goal goal;
      goal.confirmation_token = "E";
      rclcpp_action::Client<ExecuteMotion>::SendGoalOptions options;
      status_->setText("Execution requested.");
      options.goal_response_callback = [this](auto goal_handle) {
        status_->setText(goal_handle ? "Execution accepted." : "Execution rejected.");
      };
      options.feedback_callback = [this](auto, const auto feedback) {
        status_->setText(QString::fromStdString("Executing: " + feedback->active_segment));
      };
      options.result_callback = [this](const auto & wrapped_result) {
        status_->setText(QString::fromStdString(wrapped_result.result->message));
      };
      execute_client_->async_send_goal(goal, options);
    });
  }

  void onInitialize() override
  {
    auto rviz_node = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
    preview_client_ = rclcpp_action::create_client<PreviewMotion>(
      rviz_node,
      "/sonopet/preview_motion");
    execute_client_ = rclcpp_action::create_client<ExecuteMotion>(
      rviz_node,
      "/sonopet/execute_motion");
    preview_button_->setEnabled(true);
    execute_button_->setEnabled(true);
    status_->setText("Motion controls ready.");
  }

private:
  using PreviewMotion = fr3_sonopet_interfaces::action::PreviewMotion;
  using ExecuteMotion = fr3_sonopet_interfaces::action::ExecuteMotion;

  QPushButton * preview_button_;
  QPushButton * execute_button_;
  QLineEdit * confirmation_;
  QLabel * status_;
  rclcpp_action::Client<PreviewMotion>::SharedPtr preview_client_;
  rclcpp_action::Client<ExecuteMotion>::SharedPtr execute_client_;
};

}  // namespace fr3_sonopet_interfaces

PLUGINLIB_EXPORT_CLASS(fr3_sonopet_interfaces::MotionControlPanel, rviz_common::Panel)
