<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$userid = $_SESSION[check_valid_user()];
$my_profile = new profile($userid);
$_SESSION['my_profile'] = $my_profile;
$my_profile->display_profile();

do_footer();
?>